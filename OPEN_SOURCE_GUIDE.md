# 🚀 Knowledge Brain 开源发布指南

本指南将帮助你完成项目的开源发布流程。

## ✅ 发布前检查清单

### 1. 文件准备
- [x] `README.md` - 项目主文档（已创建）
- [x] `LICENSE` - MIT 开源许可证（已创建）
- [x] `CONTRIBUTING.md` - 贡献指南（已创建）
- [x] `.gitignore` - Git 忽略配置（已更新）

### 2. 敏感信息检查
确保以下信息**不会**被提交：
- [ ] `.env` 文件中的 API 密钥
- [ ] 数据库密码
- [ ] MinIO 访问密钥
- [ ] 个人身份信息

运行以下命令检查：
```bash
# 查看 .env 是否在 .gitignore 中
cat .gitignore | grep .env

# 应该看到 .env 被列出
```

### 3. 代码清理
```bash
# 删除临时文件
Remove-Item -Recurse -Force __pycache__ -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force .pytest_cache -ErrorAction SilentlyContinue

# 检查是否有大文件（模型文件等）
git status
```

## 📦 发布步骤

### 步骤 1: 提交所有更改

```bash
# 添加所有开源相关文件
git add README.md LICENSE CONTRIBUTING.md .gitignore

# 提交
git commit -m "docs: 添加开源许可证和项目文档

- 添加 MIT 开源许可证
- 添加项目 README 文档
- 添加贡献指南
- 更新 .gitignore 配置"
```

### 步骤 2: 推送到 GitHub

```bash
# 推送到 GitHub 仓库
git push github master
```

### 步骤 3: 在 GitHub 上完善项目信息

访问 https://github.com/Qw462/knowledge_brain

1. **添加项目描述**
   - 进入 Settings → General
   - 在 "Description" 填写：
     ```
     基于 RAG 与多策略检索融合的智能知识大脑系统，支持多路召回、RRF 融合排序和 Ragas 量化评估
     ```

2. **添加项目主题标签**
   - 在仓库首页点击 "About" 区域的齿轮图标
   - 添加 Topics：
     - `rag`
     - `langchain`
     - `langgraph`
     - `vector-database`
     - `milvus`
     - `bge-m3`
     - `fastapi`
     - `python`
     - `ai-agent`
     - `question-answering`

3. **设置网站链接**（可选）
   - 如果有在线演示或文档站点

### 步骤 4: 创建 GitHub Release

```bash
# 创建标签
git tag -a v1.0.0 -m "v1.0.0: 首个开源版本

主要特性：
- 基于 LangGraph 的智能工作流编排
- 多策略检索融合（向量+HyDE+联网）
- RRF 融合排序 + Reranker 重排
- Ragas 五维量化评估
- FastAPI Web 服务
- 支持 PDF/Markdown 文档导入"

# 推送标签
git push github v1.0.0
```

然后在 GitHub 上：
1. 进入 Releases 页面
2. 点击 "Draft a new release"
3. 选择刚创建的 tag
4. 填写 Release Notes（可以复制上面的信息）
5. 点击 "Publish release"

### 步骤 5: 同步到 Gitee（可选）

```bash
# 推送到 Gitee
git push origin master
git push origin v1.0.0
```

## 📢 开源后推广

### 1. 社交媒体分享
- 微博/知乎/掘金/CSDN 发布项目介绍文章
- 包含项目链接和技术亮点

### 2. 技术社区
- 在相关技术论坛发帖
- 参加开源项目展示活动

### 3. README 中添加徽章

在 README.md 开头添加：

```markdown
[![GitHub stars](https://img.shields.io/github/stars/Qw462/knowledge_brain.svg?style=social)](https://github.com/Qw462/knowledge_brain/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/Qw462/knowledge_brain.svg?style=social)](https://github.com/Qw462/knowledge_brain/network)
[![GitHub issues](https://img.shields.io/github/issues/Qw462/knowledge_brain.svg)](https://github.com/Qw462/knowledge_brain/issues)
[![GitHub pull requests](https://img.shields.io/github/issues-pr/Qw462/knowledge_brain.svg)](https://github.com/Qw462/knowledge_brain/pulls)
```

## 🔧 维护建议

### 1. 及时响应 Issues
- 设置 GitHub 通知
- 24 小时内回复 Issues
- 使用标签管理 Issues（bug、enhancement、question）

### 2. 代码审查
- 对 Pull Request 进行代码审查
- 确保代码质量
- 感谢贡献者

### 3. 定期更新
- 修复 Bug
- 添加新功能
- 更新文档
- 发布新版本

### 4. 社区建设
- 欢迎贡献者
- 维护讨论区
- 组织线上/线下活动

## 📊 数据监控

关注以下指标：
- ⭐ Stars 数量
- 🍴 Forks 数量
- 👀 Watchers 数量
- 📥 下载量
- 🐛 Issues 数量
- 🔀 Pull Requests 数量

## ⚠️ 注意事项

1. **保持代码质量** - 开源后会有更多人查看代码
2. **文档完整性** - 确保 README 清晰易懂
3. **安全性** - 不要提交敏感信息
4. **许可证** - 保留原作者的版权声明
5. **版本管理** - 使用语义化版本号（SemVer）

## 🎉 完成！

完成以上步骤后，你的项目就正式开源了！

记得在 README 中添加感谢语：

```markdown
## 🙏 致谢

感谢所有贡献者的支持！

<a href="https://github.com/Qw462/knowledge_brain/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Qw462/knowledge_brain" />
</a>
```

---

**祝你的开源项目成功！** 🚀
