"""
性能基准测试

测试内容：
1. 推理速度
2. 训练吞吐量
3. 内存使用
4. 不同模型规模对比
5. 批量大小影响

注意：这些测试标记为 'slow'，默认不运行
使用 --runslow 参数运行
"""

import pytest
import time
import torch
import gc

from model import GPT, GPTConfig
from tokenizer import BPETokenizer
from generate import TextGenerator


@pytest.mark.slow
class TestInferenceSpeed:
    """推理速度基准测试"""

    @pytest.fixture
    def mini_model(self):
        config = GPTConfig(
            vocab_size=100,
            emb_dim=64,
            num_heads=4,
            num_layers=2,
            context_size=64,
            dropout=0.0
        )
        model = GPT(config)
        model.eval()
        return model

    @pytest.fixture
    def tokenizer(self):
        tokenizer = BPETokenizer(vocab_size=100)
        tokenizer.fit(["测 试 文 本", "速 度 基 准"], verbose=False)
        return tokenizer

    @pytest.fixture
    def generator(self, mini_model, tokenizer):
        return TextGenerator(mini_model, tokenizer, torch.device("cpu"))

    def test_single_token_generation_speed(self, generator):
        """测试单个 token 生成速度"""
        prompt = "测"

        start = time.time()
        result = generator.generate(prompt, max_length=10, temperature=1.0)
        elapsed = time.time() - start

        # 应该在合理时间内完成
        assert elapsed < 10.0  # 10秒内完成
        assert len(result) > 0

        print(f"  生成速度: {elapsed:.3f}秒")

    def test_batch_generation_speed(self, generator):
        """测试批量生成速度"""
        prompts = ["测"] * 5

        start = time.time()
        results = [generator.generate(p, max_length=10) for p in prompts]
        elapsed = time.time() - start

        assert len(results) == 5
        assert elapsed < 30.0

        print(f"  批量生成速度: {elapsed:.3f}秒 ({len(prompts)/elapsed:.1f} 样本/秒)")

    def test_temperature_impact(self, generator):
        """测试温度对速度的影响"""
        prompt = "测"

        temperatures = [0.1, 0.8, 1.0, 1.5]
        times = []

        for temp in temperatures:
            start = time.time()
            generator.generate(prompt, max_length=20, temperature=temp)
            elapsed = time.time() - start
            times.append(elapsed)

        # 温度不应该显著影响速度
        max_time_diff = max(times) - min(times)
        assert max_time_diff < 2.0  # 差异小于2秒

        print(f"  温度影响: 最快 {min(times):.3f}s, 最慢 {max(times):.3f}s")


@pytest.mark.slow
class TestTrainingThroughput:
    """训练吞吐量基准测试"""

    def test_training_step_throughput(self):
        """测试训练步骤吞吐量"""
        config = GPTConfig(
            vocab_size=200,
            emb_dim=64,
            num_heads=4,
            num_layers=2,
            context_size=64,
            dropout=0.0
        )
        model = GPT(config)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        batch_size = 8
        seq_len = 32

        input_ids = torch.randint(0, 200, (batch_size, seq_len))
        target_ids = torch.randint(0, 200, (batch_size, seq_len))

        # 预热
        for _ in range(3):
            logits, loss = model(input_ids, target_ids)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 测量
        num_steps = 20
        start = time.time()

        for _ in range(num_steps):
            logits, loss = model(input_ids, target_ids)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        elapsed = time.time() - start
        throughput = (num_steps * batch_size) / elapsed

        print(f"  训练吞吐量: {throughput:.1f} 样本/秒")
        print(f"  每步时间: {elapsed/num_steps*1000:.1f}ms")

        assert throughput > 1.0  # 至少 1 样本/秒

    def test_different_batch_sizes(self):
        """测试不同批量大小的影响"""
        config = GPTConfig(
            vocab_size=100,
            emb_dim=32,
            num_heads=2,
            num_layers=2,
            context_size=32,
            dropout=0.0
        )
        model = GPT(config)
        model.train()

        seq_len = 16
        batch_sizes = [1, 2, 4, 8]
        times = []

        for bs in batch_sizes:
            input_ids = torch.randint(0, 100, (bs, seq_len))
            target_ids = torch.randint(0, 100, (bs, seq_len))

            # 预热
            for _ in range(2):
                model(input_ids, target_ids)

            # 测量
            start = time.time()
            for _ in range(10):
                model(input_ids, target_ids)
            elapsed = time.time() - start
            times.append(elapsed)

        # 批量处理应该更高效
        # 虽然大批次单次慢，但每样本时间应该更短
        single_sample_time = times[0] / (10 * 1)
        batch_sample_time = times[-1] / (10 * batch_sizes[-1])

        print(f"  单样本时间 (bs=1): {single_sample_time*1000:.1f}ms")
        print(f"  单样本时间 (bs={batch_sizes[-1]}): {batch_sample_time*1000:.1f}ms")


@pytest.mark.slow
class TestMemoryUsage:
    """内存使用基准测试"""

    def test_model_memory(self):
        """测试模型内存占用"""
        config = GPTConfig(
            vocab_size=500,
            emb_dim=128,
            num_heads=4,
            num_layers=4,
            context_size=128,
            dropout=0.0
        )

        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # 创建模型前
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            start_mem = torch.cuda.memory_allocated()

        model = GPT(config)

        # 创建模型后
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            end_mem = torch.cuda.memory_allocated()
            model_mem_mb = (end_mem - start_mem) / (1024 * 1024)
            print(f"  模型显存占用: {model_mem_mb:.1f} MB")
        else:
            param_size = sum(p.numel() * p.element_size() for p in model.parameters())
            print(f"  模型参数大小: {param_size / (1024 * 1024):.1f} MB")

    def test_forward_pass_memory(self):
        """测试前向传播内存占用"""
        config = GPTConfig(
            vocab_size=200,
            emb_dim=64,
            num_heads=4,
            num_layers=2,
            context_size=64,
            dropout=0.0
        )
        model = GPT(config)
        model.eval()

        batch_size = 4
        seq_len = 32

        gc.collect()

        input_ids = torch.randint(0, 200, (batch_size, seq_len))

        with torch.no_grad():
            logits, _ = model(input_ids)

        # 输出应该正确
        assert logits.shape == (batch_size, seq_len, 200)


@pytest.mark.slow
class TestModelScaling:
    """模型规模对比测试"""

    def test_different_model_sizes(self):
        """测试不同模型规模的性能"""
        configs = [
            # tiny
            GPTConfig(vocab_size=100, emb_dim=32, num_heads=2, num_layers=1, context_size=32, dropout=0.0),
            # small
            GPTConfig(vocab_size=100, emb_dim=64, num_heads=4, num_layers=2, context_size=64, dropout=0.0),
        ]

        results = []

        for cfg in configs:
            model = GPT(cfg)
            model.eval()

            # 测试推理时间
            input_ids = torch.randint(0, 100, (1, 16))

            # 预热
            for _ in range(3):
                with torch.no_grad():
                    model(input_ids)

            # 测量
            num_runs = 20
            start = time.time()

            with torch.no_grad():
                for _ in range(num_runs):
                    model(input_ids)

            elapsed = time.time() - start

            num_params = model.get_num_params()

            results.append({
                'emb_dim': cfg.emb_dim,
                'num_layers': cfg.num_layers,
                'num_params': num_params,
                'time_per_run': elapsed / num_runs * 1000  # ms
            })

        # 打印结果
        print("\n  模型规模对比:")
        print("  " + "-" * 60)
        print(f"  {'配置':<15} {'参数量':<12} {'单次推理':<12}")
        print("  " + "-" * 60)

        for r in results:
            config_str = f"{r['emb_dim']}d×{r['num_layers']}l"
            print(f"  {config_str:<15} {r['num_params']:<12} {r['time_per_run']:.2f}ms")

        # 更多参数应该需要更多时间
        assert results[0]['time_per_run'] < results[1]['time_per_run']


@pytest.mark.slow
class TestSequenceLengthImpact:
    """序列长度影响测试"""

    def test_different_sequence_lengths(self):
        """测试不同序列长度的影响"""
        config = GPTConfig(
            vocab_size=100,
            emb_dim=32,
            num_heads=2,
            num_layers=2,
            context_size=128,
            dropout=0.0
        )
        model = GPT(config)
        model.eval()

        seq_lengths = [8, 16, 32, 64]
        times = []

        for seq_len in seq_lengths:
            input_ids = torch.randint(0, 100, (1, seq_len))

            # 预热
            for _ in range(3):
                with torch.no_grad():
                    model(input_ids)

            # 测量
            num_runs = 20
            start = time.time()

            with torch.no_grad():
                for _ in range(num_runs):
                    model(input_ids)

            elapsed = time.time() - start
            times.append(elapsed / num_runs * 1000)  # ms

        # 打印结果
        print("\n  序列长度影响:")
        print("  " + "-" * 30)
        print(f"  {'长度':<10} {'时间':<10}")
        print("  " + "-" * 30)

        for length, time_ms in zip(seq_lengths, times):
            print(f"  {length:<10} {time_ms:.2f}ms")

        # 更长的序列应该需要更多时间
        assert times[0] < times[-1]


@pytest.mark.slow
class TestTokenizerSpeed:
    """分词器速度测试"""

    def test_encoding_speed(self):
        """测试编码速度"""
        tokenizer = BPETokenizer(vocab_size=500)
        texts = ["这是一个测试文本"] * 100

        tokenizer.fit(texts[:10], verbose=False)

        # 预热
        for text in texts[:10]:
            tokenizer.encode(text)

        # 测量
        start = time.time()
        for text in texts:
            tokenizer.encode(text)
        elapsed = time.time() - start

        throughput = len(texts) / elapsed
        print(f"  编码速度: {throughput:.0f} 文本/秒")

        assert throughput > 10  # 至少 10 文本/秒

    def test_decoding_speed(self):
        """测试解码速度"""
        tokenizer = BPETokenizer(vocab_size=200)
        texts = ["测试 文本"]
        tokenizer.fit(texts, verbose=False)

        # 创建一些 token IDs
        token_ids = [i % len(tokenizer.vocab) for i in range(1000)]

        # 预热
        for _ in range(10):
            tokenizer.decode(token_ids[:50])

        # 测量
        num_runs = 100
        start = time.time()

        for _ in range(num_runs):
            tokenizer.decode(token_ids)

        elapsed = time.time() - start
        throughput = (num_runs * len(token_ids)) / elapsed

        print(f"  解码速度: {throughput:.0f} tokens/秒")

        assert throughput > 1000  # 至少 1000 tokens/秒


@pytest.mark.slow
@pytest.mark.gpu
class TestGPUBenchmarks:
    """GPU 基准测试（仅在 GPU 可用时运行）"""

    def test_gpu_vs_cpu_speed(self):
        """测试 GPU vs CPU 速度"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA 不可用")

        config = GPTConfig(
            vocab_size=200,
            emb_dim=64,
            num_heads=4,
            num_layers=2,
            context_size=64,
            dropout=0.0
        )

        # CPU 测试
        model_cpu = GPT(config)
        model_cpu.eval()

        input_ids = torch.randint(0, 200, (4, 32))

        # CPU 预热
        for _ in range(5):
            with torch.no_grad():
                model_cpu(input_ids)

        # CPU 测量
        num_runs = 20
        start = time.time()

        with torch.no_grad():
            for _ in range(num_runs):
                model_cpu(input_ids)

        cpu_time = (time.time() - start) / num_runs

        # GPU 测试
        model_gpu = GPT(config).cuda()
        model_gpu.eval()

        input_ids_gpu = torch.randint(0, 200, (4, 32)).cuda()

        # GPU 预热
        for _ in range(5):
            with torch.no_grad():
                model_gpu(input_ids_gpu)

        torch.cuda.synchronize()

        # GPU 测量
        start = time.time()

        with torch.no_grad():
            for _ in range(num_runs):
                model_gpu(input_ids_gpu)

        torch.cuda.synchronize()
        gpu_time = (time.time() - start) / num_runs

        print(f"\n  CPU 时间: {cpu_time*1000:.2f}ms")
        print(f"  GPU 时间: {gpu_time*1000:.2f}ms")
        print(f"  加速比: {cpu_time/gpu_time:.1f}x")

        # GPU 应该更快（但允许一定误差）
        assert gpu_time < cpu_time * 2  # 至少接近或更快
