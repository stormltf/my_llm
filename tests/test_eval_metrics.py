"""
评估指标测试

测试内容：
1. BLEU 分数计算
2. ROUGE 分数计算
3. 困惑度计算细节
4. 准确率评估
5. 多维度评估
"""

import pytest
import torch
import math

from model import GPT, GPTConfig
from tokenizer import BPETokenizer
from generate import TextGenerator


class TestPerplexityMetrics:
    """困惑度指标测试"""

    @pytest.fixture
    def model_and_tokenizer(self):
        config = GPTConfig(
            vocab_size=100,
            emb_dim=32,
            num_heads=2,
            num_layers=2,
            context_size=32,
            dropout=0.0
        )
        model = GPT(config)
        model.eval()

        tokenizer = BPETokenizer(vocab_size=100)
        tokenizer.fit(["测 试 评 估 指 标", "困 难 度 计 算"], verbose=False)

        return model, tokenizer

    @pytest.fixture
    def generator(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        return TextGenerator(model, tokenizer, torch.device("cpu"))

    def test_perplexity_calculation(self, generator):
        """测试困惑度计算"""
        text = "测 试"
        ppl = generator.get_perplexity(text)

        # 困惑度应该大于 1
        assert ppl > 1.0
        # 困惑度应该是有限数
        assert ppl < float('inf')
        # 不应该是 NaN
        assert not math.isnan(ppl)

    def test_perplexity_same_text(self, generator):
        """测试相同文本的困惑度一致性"""
        text = "测 试 评 估"

        ppl1 = generator.get_perplexity(text)
        ppl2 = generator.get_perplexity(text)

        # 相同文本应该有相同困惑度
        assert ppl1 == ppl2

    def test_perplexity_different_texts(self, generator):
        """测试不同文本可能有不同困惑度"""
        text1 = "测 试"
        text2 = "指 标"

        ppl1 = generator.get_perplexity(text1)
        ppl2 = generator.get_perplexity(text2)

        # 都是有效困惑度
        assert ppl1 > 0
        assert ppl2 > 0

    def test_perplexity_empty_text(self, generator):
        """测试空文本困惑度"""
        # 空文本可能导致问题
        # 这取决于实现
        import math
        try:
            ppl = generator.get_perplexity("")
            # 如果没有报错，检查返回值
            # 空文本可能返回 NaN 或 inf，这也是合理的
            assert ppl > 0 or math.isnan(ppl) or math.isinf(ppl)
        except (IndexError, ValueError):
            # 空文本报错也是合理的
            pass

    def test_perplexity_comparison(self):
        """测试困惑度对比逻辑"""
        # 创建两个模型：一个随机初始化，一个经过"训练"
        config = GPTConfig(
            vocab_size=50,
            emb_dim=32,
            num_heads=2,
            num_layers=2,
            context_size=32,
            dropout=0.0
        )

        # 未训练模型
        model_untrained = GPT(config)
        model_untrained.eval()

        # "训练"后的模型（简单几步）
        model_trained = GPT(config)
        model_trained.train()

        optimizer = torch.optim.Adam(model_trained.parameters(), lr=1e-3)
        for _ in range(10):
            input_ids = torch.randint(0, 50, (4, 16))
            target_ids = torch.randint(0, 50, (4, 16))
            logits, loss = model_trained(input_ids, target_ids)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model_trained.eval()

        tokenizer = BPETokenizer(vocab_size=50)
        tokenizer.fit(["测 试 文 本"], verbose=False)

        gen_untrained = TextGenerator(model_untrained, tokenizer, torch.device("cpu"))
        gen_trained = TextGenerator(model_trained, tokenizer, torch.device("cpu"))

        text = "测 试"
        ppl_untrained = gen_untrained.get_perplexity(text)
        ppl_trained = gen_trained.get_perplexity(text)

        # 都是有效困惑度
        assert ppl_untrained > 1.0
        assert ppl_trained > 0


class TestBLEUMetrics:
    """BLEU 指标测试（简化实现）"""

    def test_bleu_score_exact_match(self):
        """测试完全匹配的 BLEU 分数"""
        # 简化的 BLEU 计算
        reference = "这 是 一 个 测 试"
        candidate = "这 是 一 个 测 试"

        # 完全匹配应该得到最高分
        # 简单计算：词级别的匹配率
        ref_words = reference.split()
        cand_words = candidate.split()

        matches = sum(1 for w in cand_words if w in ref_words)
        precision = matches / len(cand_words) if cand_words else 0

        assert precision == 1.0

    def test_bleu_score_no_match(self):
        """测试完全不匹配的 BLEU 分数"""
        reference = "这 是 测 试"
        candidate = "完 全 不 同 的 词"

        ref_words = reference.split()
        cand_words = candidate.split()

        matches = sum(1 for w in cand_words if w in ref_words)
        precision = matches / len(cand_words) if cand_words else 0

        assert precision == 0.0

    def test_bleu_score_partial_match(self):
        """测试部分匹配的 BLEU 分数"""
        reference = "这 是 一 个 测 试"
        candidate = "这 是 另 一 个 测 试"

        ref_words = reference.split()
        cand_words = candidate.split()

        matches = sum(1 for w in cand_words if w in ref_words)
        precision = matches / len(cand_words) if cand_words else 0

        # 部分匹配
        assert 0 < precision < 1.0

    def test_bleu_score_n_gram(self):
        """测试 N-gram BLEU 分数"""
        reference = "这 是 一 个 测 试"
        candidate = "这 是 测 试"

        # 2-gram 匹配
        def get_ngrams(tokens, n):
            return [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

        ref_words = reference.split()
        cand_words = candidate.split()

        ref_bigrams = set(get_ngrams(ref_words, 2))
        cand_bigrams = set(get_ngrams(cand_words, 2))

        matches = len(ref_bigrams & cand_bigrams)
        total = len(cand_bigrams)

        precision = matches / total if total > 0 else 0

        # 应该有匹配
        assert precision > 0


class TestROUGEMetrics:
    """ROUGE 指标测试（简化实现）"""

    def test_rouge_l_recall(self):
        """测试 ROUGE-L 召回率"""
        reference = "这 是 一 个 测 试 句 子"
        candidate = "这 是 测 试"

        # 简化版：最长公共子序列
        def lcs_length(ref, cand):
            m, n = len(ref), len(cand)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if ref[i-1] == cand[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            return dp[m][n]

        ref_words = reference.split()
        cand_words = candidate.split()

        lcs = lcs_length(ref_words, cand_words)
        recall = lcs / len(ref_words) if ref_words else 0

        # 召回率应该在 0-1 之间
        assert 0 <= recall <= 1.0

    def test_rouge_l_precision(self):
        """测试 ROUGE-L 精确率"""
        reference = "这 是 测 试"
        candidate = "这 是 一 个 测 试 句 子"

        def lcs_length(ref, cand):
            m, n = len(ref), len(cand)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if ref[i-1] == cand[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            return dp[m][n]

        ref_words = reference.split()
        cand_words = candidate.split()

        lcs = lcs_length(ref_words, cand_words)
        precision = lcs / len(cand_words) if cand_words else 0

        # 精确率应该在 0-1 之间
        assert 0 <= precision <= 1.0

    def test_rouge_l_f1(self):
        """测试 ROUGE-L F1 分数"""
        reference = "这 是 一 个 测 试"
        candidate = "这 是 测 试"

        def lcs_length(ref, cand):
            m, n = len(ref), len(cand)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if ref[i-1] == cand[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            return dp[m][n]

        ref_words = reference.split()
        cand_words = candidate.split()

        lcs = lcs_length(ref_words, cand_words)
        precision = lcs / len(cand_words) if cand_words else 0
        recall = lcs / len(ref_words) if ref_words else 0

        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # F1 应该在 0-1 之间
        assert 0 <= f1 <= 1.0


class TestAccuracyMetrics:
    """准确率指标测试"""

    def test_token_level_accuracy(self):
        """测试 token 级别准确率"""
        predictions = torch.tensor([1, 2, 3, 4, 5])
        targets = torch.tensor([1, 2, 0, 4, 6])

        correct = (predictions == targets).sum().item()
        accuracy = correct / len(predictions)

        # 3/5 = 0.6
        assert accuracy == 0.6

    def test_sequence_accuracy(self):
        """测试序列准确率"""
        predictions = [
            torch.tensor([1, 2, 3]),
            torch.tensor([4, 5, 6]),
            torch.tensor([7, 8, 9])
        ]
        targets = [
            torch.tensor([1, 2, 3]),
            torch.tensor([4, 0, 6]),
            torch.tensor([7, 8, 9])
        ]

        correct = sum(
            1 for p, t in zip(predictions, targets)
            if torch.equal(p, t)
        )
        accuracy = correct / len(predictions)

        # 2/3 完全正确
        assert accuracy == 2/3

    def test_top_k_accuracy(self):
        """测试 Top-k 准确率"""
        # 模拟 logits
        logits = torch.tensor([
            [0.1, 0.2, 5.0, 0.3],  # 真实是 2
            [0.5, 3.0, 0.1, 0.2],  # 真实是 1
            [2.0, 0.1, 0.2, 3.0],  # 真实是 3
        ])
        targets = torch.tensor([2, 1, 3])

        # Top-1 准确率
        top1_pred = logits.argmax(dim=-1)
        top1_acc = (top1_pred == targets).float().mean().item()

        assert top1_acc == 1.0

        # Top-2 准确率
        top2_pred = torch.topk(logits, k=2, dim=-1).indices
        top2_acc = sum(
            1 for i in range(len(targets))
            if targets[i] in top2_pred[i]
        ) / len(targets)

        assert top2_acc == 1.0


class TestMultiDimMetrics:
    """多维度评估测试"""

    def test_combined_evaluation(self):
        """测试组合评估"""
        # 模拟生成结果
        generations = [
            {"reference": "你好 世界", "candidate": "你好 世界"},
            {"reference": "测试 文本", "candidate": "测 试 文 本"},
            {"reference": "评估 指标", "candidate": "评 估 标"},
        ]

        results = []
        for item in generations:
            ref = item["reference"]
            cand = item["candidate"]

            # 词级匹配率
            ref_words = ref.split()
            cand_words = cand.split()
            matches = sum(1 for w in cand_words if w in ref_words)
            precision = matches / len(cand_words) if cand_words else 0

            results.append(precision)

        # 平均精度
        avg_precision = sum(results) / len(results)

        # 应该在合理范围内
        assert 0 <= avg_precision <= 1.0

    def test_length_penalty(self):
        """测试长度惩罚"""
        reference = "这 是 一 个 测 试 句 子"
        candidate_short = "这 是 测 试"
        candidate_long = "这 是 一 个 非 常 长 的 测 试 句 子 有 很 多 额 外 的 词"

        ref_len = len(reference.split())

        # 简化的长度惩罚
        def length_penalty(ref_len, cand_len):
            ratio = cand_len / ref_len if ref_len > 0 else 0
            if ratio < 1.0:
                return ratio
            else:
                return 1.0

        penalty_short = length_penalty(ref_len, len(candidate_short.split()))
        penalty_long = length_penalty(ref_len, len(candidate_long.split()))

        # 短的惩罚，长的不惩罚
        assert penalty_short < 1.0
        assert penalty_long == 1.0


class TestEvaluationUtilities:
    """评估工具函数测试"""

    def test_create_evaluation_report(self):
        """测试生成评估报告"""
        metrics = {
            "perplexity": 15.5,
            "accuracy": 0.85,
            "bleu": 0.72,
            "rouge_l": 0.68
        }

        report = []
        report.append("=" * 40)
        report.append("评估报告")
        report.append("=" * 40)

        for name, value in metrics.items():
            report.append(f"{name}: {value:.4f}")

        report_text = "\n".join(report)

        assert "perplexity: 15.5000" in report_text
        assert "accuracy: 0.8500" in report_text
        assert "bleu: 0.7200" in report_text
        assert "rouge_l: 0.6800" in report_text

    def test_compare_models(self):
        """测试模型对比"""
        model_a_metrics = {"accuracy": 0.80, "perplexity": 20.0}
        model_b_metrics = {"accuracy": 0.85, "perplexity": 15.0}

        # 模型 B 应该更好
        assert model_b_metrics["accuracy"] > model_a_metrics["accuracy"]
        assert model_b_metrics["perplexity"] < model_a_metrics["perplexity"]

    def test_aggregate_metrics(self):
        """测试指标聚合"""
        metrics_list = [
            {"loss": 2.5, "accuracy": 0.7},
            {"loss": 2.0, "accuracy": 0.75},
            {"loss": 1.8, "accuracy": 0.8},
            {"loss": 1.5, "accuracy": 0.82},
        ]

        avg_loss = sum(m["loss"] for m in metrics_list) / len(metrics_list)
        avg_acc = sum(m["accuracy"] for m in metrics_list) / len(metrics_list)

        assert avg_loss == 1.95
        assert avg_acc == 0.7675


class TestBatchEvaluation:
    """批量评估测试"""

    def test_batch_perplexity(self):
        """测试批量困惑度计算"""
        config = GPTConfig(
            vocab_size=50,
            emb_dim=32,
            num_heads=2,
            num_layers=2,
            context_size=32,
            dropout=0.0
        )
        model = GPT(config)
        model.eval()

        tokenizer = BPETokenizer(vocab_size=50)
        tokenizer.fit(["测 试 评 估"], verbose=False)

        generator = TextGenerator(model, tokenizer, torch.device("cpu"))

        texts = ["测 试", "评 估", "指 标"]
        ppls = [generator.get_perplexity(text) for text in texts]

        # 所有困惑度应该有效
        assert all(ppl > 0 for ppl in ppls)
        assert all(ppl < float('inf') for ppl in ppls)

    def test_evaluation_statistics(self):
        """测试评估统计"""
        perplexities = [10.5, 15.2, 12.8, 20.1, 11.3]

        import statistics

        mean_ppl = statistics.mean(perplexities)
        median_ppl = statistics.median(perplexities)
        stdev_ppl = statistics.stdev(perplexities)

        # 检查统计值合理
        assert mean_ppl > 0
        assert median_ppl > 0
        assert stdev_ppl >= 0
