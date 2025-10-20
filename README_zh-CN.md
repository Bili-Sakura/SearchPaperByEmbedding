[English](./README.md) | [简体中文](./README_zh-CN.md)

# 🔍 ICLR 2026 论文搜索

使用语义相似度搜索 18,000+ 篇 ICLR 2026 投稿论文。

## 这是什么？

一个简单的搜索工具，通过自然语言描述来查找研究论文。只需输入你的查询，即可获得相关论文！

## 功能特点

- 🔎 **自然语言搜索** - 用简单的话描述论文
- ⚡ **即时结果** - 预计算嵌入向量，搜索快速
- 🎯 **智能筛选** - 按研究领域过滤
- 📊 **18,000+ 论文** - 所有 ICLR 2026 投稿
- 🆓 **免费开源**

## 使用方法

1. **输入搜索查询** - 描述你想找的论文
2. **调整设置**（可选）- 结果数量、研究领域过滤
3. **点击搜索** - 获得按相关度排序的论文

### 示例查询

- "图像分类的视觉 Transformer"
- "长序列的高效注意力机制"
- "元学习的小样本学习"
- "图像生成的扩散模型"
- "分子性质预测的图神经网络"

## 本地运行

```bash
# 安装依赖
pip install -r requirements.txt

# 运行应用
python app.py
```

访问 `http://localhost:7860`

## 部署到 Hugging Face Spaces

1. 在 https://huggingface.co/spaces 创建新 Space
2. 推送代码:
   ```bash
   git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/SPACE_NAME
   git push hf main
   ```
3. 你的搜索工具即刻上线！

**注意**: 确保上传 `iclr2026_papers.json` 和缓存文件 `output/cache_*.npy` 到 Space。

## 工作原理

论文被转换为嵌入向量（数值向量），捕捉其语义含义。当你搜索时，查询被转换为相同格式，我们使用余弦相似度找出最相似向量的论文。

## 技术栈

- **框架**: Gradio
- **嵌入模型**: all-MiniLM-L6-v2 (快速, 384 维)
- **数据集**: OpenReview 的 ICLR 2026 投稿

## 引用

```bibtex
@misc{SearchPaperByEmbedding,
  author = {gyj155},
  title = {ICLR 2026 Paper Search},
  year = {2025},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/gyj155/SearchPaperByEmbedding}}
}
```

## 许可证

MIT License

---

⭐ 如果这个工具帮你找到了论文，请给仓库加星！
