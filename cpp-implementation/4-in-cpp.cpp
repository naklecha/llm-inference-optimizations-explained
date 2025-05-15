#include "ATen/core/ATen_fwd.h"
#include "c10/core/ScalarType.h"
#include "c10/util/BFloat16.h"
#include "torch/types.h"
#include <ATen/ops/scaled_dot_product_attention.h>
#include <torch/torch.h>
#include <cnpy.h>
#include <iostream>

using namespace std;
namespace cfg {
    constexpr int64_t dim           = 2048;
    constexpr int64_t n_layers      = 16;
    constexpr int64_t n_heads       = 32;
    constexpr int64_t n_kv_heads    = 8;
    constexpr int64_t kv_mult       = 4;
    constexpr int64_t head_dim      = 64;
    constexpr int64_t rot_dim       = 64;
    constexpr double  norm_eps      = 1e-5;
    constexpr double  rope_theta    = 1'000'000.0;
    constexpr int64_t max_new       = 500;
}

inline torch::Tensor rms_norm(const torch::Tensor& x, const torch::Tensor& w, double eps = cfg::norm_eps) {
    torch::Tensor rms = torch::rsqrt((x.square().mean(-1, true)) + eps);
    return (x * rms) * w;
}

torch::Tensor get_freqs_cis(int64_t max_tokens, double rope_theta, torch::Device device) {
    torch::Tensor idx  = torch::arange(cfg::rot_dim / 2) / (cfg::rot_dim / 2);
    torch::Tensor positions = torch::arange(max_tokens).unsqueeze(1);
    torch::Tensor freqs = positions / torch::pow(torch::tensor(rope_theta, idx.options()), idx);
    torch::Tensor ones  = torch::ones_like(freqs);
    return torch::polar(ones, freqs).to(device);
}

torch::Tensor apply_rope(torch::Tensor& t, const torch::Tensor& freq_layer, int64_t B) {
    torch::Tensor t_rot  = t.narrow(-1, 0, cfg::rot_dim).contiguous();
    torch::Tensor t_pass = t.narrow(-1, cfg::rot_dim, cfg::head_dim - cfg::rot_dim);
    t_rot = t_rot.to(torch::kFloat).view({B, -1, cfg::rot_dim / 2, 2});
    torch::Tensor trot_c = torch::view_as_complex(t_rot) * freq_layer.unsqueeze(0).unsqueeze(0);
    torch::Tensor t_out  = torch::view_as_real(trot_c).view({B, -1, cfg::rot_dim});
    return torch::cat({t_out, t_pass}, -1).to(torch::kBFloat16);
}

torch::Tensor transformer_block_forward(torch::Tensor& x, int64_t layer, torch::OrderedDict<std::string, torch::Tensor>& model, torch::Tensor& kcache, torch::Tensor& vcache, int64_t pos, const torch::Tensor& freqs_cis, torch::Device device) {
    int64_t B = x.size(0);
    torch::Tensor W_attn_norm = model["layers." + std::to_string(layer) + ".attention_norm.weight"];
    torch::Tensor x_norm = rms_norm(x, W_attn_norm);
    torch::Tensor Wq = model["layers." + std::to_string(layer) + ".attention.wq.weight"];
    torch::Tensor Wk = model["layers." + std::to_string(layer) + ".attention.wk.weight"];
    torch::Tensor Wv = model["layers." + std::to_string(layer) + ".attention.wv.weight"];

    torch::Tensor q = torch::matmul(x_norm, Wq.t()).view({B, cfg::n_heads,  cfg::head_dim});
    torch::Tensor k = torch::matmul(x_norm, Wk.t()).view({B, cfg::n_kv_heads, cfg::head_dim});
    torch::Tensor v = torch::matmul(x_norm, Wv.t()).view({B, cfg::n_kv_heads, cfg::head_dim});


    torch::Tensor freqs_layer = freqs_cis.index({pos}).slice(/*dim=*/-1, 0, cfg::rot_dim / 2);
    q = apply_rope(q, freqs_layer, B);
    k = apply_rope(k, freqs_layer, B);

    if (cfg::kv_mult > 1) {
        k = k.repeat_interleave(cfg::kv_mult, 1);
        v = v.repeat_interleave(cfg::kv_mult, 1);
    }

    kcache.index({layer, pos}).copy_(k);
    vcache.index({layer, pos}).copy_(v);

    torch::Tensor k_all = kcache.index({layer,torch::indexing::Slice(0, pos + 1)}).permute({1, 2, 0, 3});
    torch::Tensor v_all = vcache.index({layer,torch::indexing::Slice(0, pos + 1)}).permute({1, 2, 0, 3});
    

    torch::Tensor q_  = q.unsqueeze(2);

    torch::Tensor out = at::scaled_dot_product_attention(
        q_, 
        k_all, 
        v_all,
        c10::nullopt,
        false,
        false,
        1.0 / std::sqrt(cfg::head_dim)
    ).squeeze(2);

    out = out.reshape({B, cfg::dim});

    torch::Tensor Wo  = model["layers." + std::to_string(layer) + ".attention.wo.weight"].to(torch::kBFloat16);

    x = x + torch::matmul(out, Wo.t());

    torch::Tensor W_ffn_norm = model["layers." + std::to_string(layer) + ".ffn_norm.weight"];
    torch::Tensor x_ffn_norm = rms_norm(x, W_ffn_norm);

    torch::Tensor W1 = model["layers." + std::to_string(layer) + ".feed_forward.w1.weight"];
    torch::Tensor W2 = model["layers." + std::to_string(layer) + ".feed_forward.w2.weight"];
    torch::Tensor W3 = model["layers." + std::to_string(layer) + ".feed_forward.w3.weight"];

    torch::Tensor ff1 = torch::matmul(x_ffn_norm, W1.t());
    torch::Tensor ff3 = torch::matmul(x_ffn_norm, W3.t());
    torch::Tensor ffn = torch::silu(ff1) * ff3;
    ffn = torch::matmul(ffn, W2.t());

    return x + ffn;
}

torch::OrderedDict<std::string, torch::Tensor> load_model(const std::string& npz_path, torch::Device device) {
    torch::NoGradGuard no_grad;
    cnpy::npz_t loaded_model = cnpy::npz_load(npz_path);
    torch::OrderedDict<std::string, torch::Tensor> model;
    for (auto& [name, arr] : loaded_model) {
        const torch::BFloat16* data_ptr = arr.data<torch::BFloat16>();
        vector<int64_t> sizes(arr.shape.begin(), arr.shape.end());
        torch::Tensor tensor = torch::from_blob(const_cast<torch::BFloat16*>(data_ptr), sizes).clone().to(torch::kBFloat16).to(device);
        model.insert(name, tensor);
    }
    return model;
}

torch::Tensor get_tokens(torch::Device device) {
    torch::NoGradGuard no_grad;
    std::vector<std::vector<int64_t>> input_data = {
        {128000,128006,882,128007,198,8144,264,2875,3446,922,264,892,63865,889,21728,14154,22463,13,128009,128006,78191,128007,198},
        {128000,128006,882,128007,198,4110,264,21976,1990,1403,15592,6067,25394,3823,21958,13,128009,128006,78191,128007,198},
        {128000,128006,882,128007,198,75885,264,80320,3363,304,279,1060,220,12112,15,13,128009,128006,78191,128007,198},
        {128000,128006,882,128007,198,8144,264,6661,505,264,16700,11323,655,311,872,3070,1203,2162,13,128009,128006,78191,128007,198},
        {128000,128006,882,128007,198,4110,264,18884,1917,449,5016,11204,6067,323,20566,13,128009,128006,78191,128007,198},
        {128000,128006,882,128007,198,849,21435,31228,25213,311,264,220,605,4771,6418,13,128009,128006,78191,128007,198},
        {128000,128006,882,128007,198,28474,323,13168,32682,323,51618,13,128009,128006,78191,128007,198},
        {128000,128006,882,128007,198,2127,56956,279,11384,323,6372,315,10182,2349,13,128009,128006,78191,128007,198},
        {128000,128006,882,128007,198,849,21435,279,1920,315,7397,74767,304,7872,13,128009,128006,78191,128007,198},
        {128000,128006,882,128007,198,23340,1523,279,32874,315,18428,5557,13,128009,128006,78191,128007,198},
        {128000,128006,882,128007,198,50,3884,4330,18699,10105,311,8108,12466,12571,13,128009,128006,78191,128007,198},
        {128000,128006,882,128007,198,21103,264,7446,14348,369,7340,26206,13,128009,128006,78191,128007,198},
        {128000,128006,882,128007,198,4110,264,15496,3197,369,4423,4560,311,8343,810,6136,6108,15657,13,128009,128006,78191,128007,198},
        {128000,128006,882,128007,198,21076,264,8446,369,264,2678,2626,311,7417,2930,9546,13,128009,128006,78191,128007,198},
        {128000,128006,882,128007,198,61369,7504,311,4048,264,502,4221,30820,13,128009,128006,78191,128007,198},
        {128000,128006,882,128007,198,849,21435,279,26431,315,279,8753,22910,304,6617,3925,13,128009,128006,78191,128007,198},
        {128000,128006,882,128007,198,75885,1268,279,3823,22852,1887,28533,19338,13,128009,128006,78191,128007,198},
        {128000,128006,882,128007,198,61369,279,1401,16565,315,7524,11692,13,128009,128006,78191,128007,198},
        {128000,128006,882,128007,198,849,21435,279,3090,11008,323,1202,12939,311,9420,596,61951,13,128009,128006,78191,128007,198},
        {128000,128006,882,128007,198,9370,5730,553,279,3682,26018,315,3823,25917,13,128009,128006,78191,128007,198},
        {128000,128006,882,128007,198,644,688,264,502,10775,430,33511,5540,315,19794,323,33819,13,128009,128006,78191,128007,198},
        {128000,128006,882,128007,198,21103,264,22556,3838,430,5829,17832,4907,13,128009,128006,78191,128007,198},
        {128000,128006,882,128007,198,4110,264,502,11363,1701,1193,14293,430,3240,449,279,6661,364,34,4527,128009,128006,78191,128007,198},
        {128000,128006,882,128007,198,21076,264,7434,369,264,2835,1847,922,958,78393,62340,13,128009,128006,78191,128007,198},
        {128000,128006,882,128007,198,52157,264,1917,1405,12966,649,7397,1910,27985,323,7664,8396,13,128009,128006,78191,128007,198},
        {128000,128006,882,128007,198,75885,264,1938,304,279,2324,315,459,14154,33589,274,3191,13,128009,128006,78191,128007,198},
        {128000,128006,882,128007,198,28474,13042,33726,9017,30084,304,2380,2204,34775,13,128009,128006,78191,128007,198},
        {128000,128006,882,128007,198,849,21435,1268,11033,5614,3728,6696,12912,13,128009,128006,78191,128007,198},
        {128000,128006,882,128007,198,2127,56956,279,5536,315,279,18991,3577,389,7665,8396,13,128009,128006,78191,128007,198},
        {128000,128006,882,128007,198,75885,279,15740,315,18273,24198,4028,3823,3925,13,128009,128006,78191,128007,198},
        {128000,128006,882,128007,198,50,3884,12823,369,18646,18547,304,1579,89561,15082,13,128009,128006,78191,128007,198},
        {128000,128006,882,128007,198,4110,264,8641,369,11469,14604,11478,13,128009,128006,78191,128007,198},
        {128000,128006,882,128007,198,61369,264,220,966,11477,8815,311,1977,264,6928,14464,13,128009,128006,78191,128007,198},
        {128000,128006,882,128007,198,61524,15174,369,7524,12324,11175,13,128009,128006,78191,128007,198},
        {128000,128006,882,128007,198,849,21435,1268,311,67632,264,6650,40543,13,128009,128006,78191,128007,198},
        {128000,128006,882,128007,198,849,21435,279,16565,4920,5780,6975,26249,13,128009,128006,78191,128007,198},
        {128000,128006,882,128007,198,75885,1268,12904,1669,6616,15207,16039,5557,4375,13,128009,128006,78191,128007,198},
        {128000,128006,882,128007,198,61369,279,1920,315,30829,264,6505,3851,13,128009,128006,78191,128007,198},
        {128000,128006,882,128007,198,849,21435,279,12624,1749,323,3493,459,3187,9526,13,128009,128006,78191,128007,198},
        {128000,128006,882,128007,198,28474,2204,4907,5942,14645,13,128009,128006,78191,128007,198},
        {128000,128006,882,128007,198,97654,279,31308,25127,315,19465,15009,304,12966,13,128009,128006,78191,128007,198},
        {128000,128006,882,128007,198,52361,279,7434,315,12437,4028,2204,41903,32006,13,128009,128006,78191,128007,198},
        {128000,128006,882,128007,198,2127,56956,279,259,75143,3575,505,5361,31308,39555,13,128009,128006,78191,128007,198},
        {128000,128006,882,128007,198,38275,279,5133,1990,5557,323,3823,23871,13,128009,128006,78191,128007,198},
        {128000,128006,882,128007,198,849,20588,279,41903,3488,315,3508,1949,690,6866,13,128009,128006,78191,128007,198},
        {128000,128006,882,128007,198,4110,264,16195,8641,311,6041,16036,60299,13,128009,128006,78191,128007,198},
        {128000,128006,882,128007,198,21076,264,4967,2068,369,264,1176,7394,45796,23055,13,128009,128006,78191,128007,198},
        {128000,128006,882,128007,198,8144,264,8982,922,279,12939,315,19071,6873,13,128009,128006,78191,128007,198},
        {128000,128006,882,128007,198,21103,264,26129,389,7524,5425,94526,10758,13,128009,128006,78191,128007,198},
        {128000,128006,882,128007,198,4110,264,50048,596,8641,311,8830,29924,4731,13,128009,128006,78191,128007,198},
    };
    int64_t max_len = 0;
    for (auto& v : input_data) max_len = std::max<int64_t>(max_len, v.size());
    cout << "max_len: " << max_len << '\n';

    int64_t B = static_cast<int64_t>(input_data.size());
    torch::Tensor tokens = torch::full(
        {B, max_len}, 
        128001,
        torch::kInt64
    ).to(device);

    for (int64_t i = 0; i < B; ++i) {
        auto& row = input_data[i];
        for (int64_t j = 0; j < static_cast<int64_t>(row.size()); ++j) {
            tokens.index_put_({i, j}, row[j]);
        }
    }

    return tokens.to(device);
}

torch::Tensor embed_tokens(torch::Tensor tokens, torch::Tensor embedding_weight, torch::Device device) {
    return torch::nn::functional::embedding(
        tokens.to(device),
        embedding_weight.to(device)
    ).to(device);
}

int main() {
    torch::NoGradGuard no_grad;
    torch::Device device(torch::kCUDA);
    
    cout << "LOADING MODEL" << '\n';
    torch::OrderedDict<std::string, torch::Tensor> model = load_model("model-weights.npz", device);
    cout << "MODEL LOADED" << '\n';

    torch::Tensor embedding_weight = model["tok_embeddings.weight"].to(device);
    int64_t vocab_size = embedding_weight.size(0);
    int64_t dim = embedding_weight.size(1);

    cout << "embedding_weight: " << embedding_weight.sizes() << '\n';

    torch::Tensor tokens = get_tokens(device);
    cout << "tokens: " << tokens.sizes() << '\n';

    const int64_t batch_size = tokens.size(0);
    const int64_t seq_len = tokens.size(1);

    const int64_t MAX_SEQ_LEN = seq_len + cfg::max_new;
    torch::Tensor freqs_cis = get_freqs_cis(MAX_SEQ_LEN, cfg::rope_theta, device);

    torch::TensorOptions options_bf16 = torch::dtype(torch::kBFloat16).device(device);
    torch::Tensor kcache = torch::empty({
        cfg::n_layers, 
        MAX_SEQ_LEN, 
        batch_size, 
        cfg::n_heads, 
        cfg::head_dim
    }, options_bf16).to(device);
    torch::Tensor vcache = torch::empty_like(kcache).to(device);

    torch::Tensor norm_weight = model["norm.weight"];
    torch::Tensor out_weight  = model["output.weight"];


    auto start_time = std::chrono::high_resolution_clock::now();
    torch::Tensor x;
    int64_t pos = 0;
    for (; pos < seq_len; ++pos) {
        torch::Tensor tok_step = tokens.index({torch::indexing::Slice(), pos});
        x = embed_tokens(tok_step, embedding_weight, device);
        for (int l = 0; l < cfg::n_layers; ++l)
            x = transformer_block_forward(x, l, model, kcache, vcache, pos, freqs_cis, device);
    }
    torch::Tensor x_last  = rms_norm(x, norm_weight);
    torch::Tensor logits  = torch::matmul(x_last, out_weight.t());
    torch::Tensor next_id = torch::argmax(logits, -1).to(torch::kInt64);


    std::vector<std::vector<int64_t>> generated(batch_size, std::vector<int64_t>(cfg::max_new));

    for (int step = 0; step < cfg::max_new; ++step, ++pos) {
        std::chrono::time_point token_start_time = std::chrono::high_resolution_clock::now();
        x = embed_tokens(next_id, embedding_weight, device).to(torch::kBFloat16);
        
        for (int l = 0; l < cfg::n_layers; ++l)
            x = transformer_block_forward(x, l, model, kcache, vcache, pos, freqs_cis, device);  

        x = rms_norm(x, norm_weight);

        logits = torch::matmul(x, out_weight.t());
        next_id = torch::argmax(logits, -1).to(torch::kInt64);

        torch::Tensor next_id_cpu = next_id.to(torch::kCPU, true);
        auto* data = next_id_cpu.data_ptr<int64_t>();
        for (int64_t b = 0; b < batch_size; ++b) generated[b][step] = data[b];

        std::chrono::duration<double> token_elapsed = std::chrono::high_resolution_clock::now() - token_start_time;
        cout << "generated: " << generated.size() << " -- time taken: " << token_elapsed.count() << " with tps: " << 50 / token_elapsed.count() << '\n';
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;


    /* ================ pretty-print results ======================== */
    for (int64_t b = 0; b < batch_size; ++b) {
        std::cout << "\nPROMPT-" << b << "  →  ";
        for (auto tk : generated[b]) std::cout << tk << ' ';
    }

    int generated_tokens = 0;
    for (int64_t b = 0; b < batch_size; ++b) {
        for (auto tk : generated[b]) {
            if (tk != 128009) {
                generated_tokens += 1;
            }
        }
    }
    std::cout << "\n\n[INFO] Generated " << generated_tokens << " tokens (greedy)\n";

    std::cout << "[INFO] Generation loop took " << elapsed.count() << " seconds.\n";
    double total_tokens = static_cast<double>(cfg::max_new * batch_size);
    double tps = total_tokens / elapsed.count();
    std::cout << "[INFO] Throughput: " << tps << " tokens/second\n";

    return 0;
}