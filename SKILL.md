# 迭代-测试-文档-上传技能 (Iteration-Test-Doc-Upload)

## 目标
在每次完成软件迭代开发后，通过完整的测试、文档更新和版本上传流程，确保新功能质量可靠、文档同步、版本控制有序。

## 适用场景
- ✅ 完成新功能开发后的验收流程
- ✅ 代码重构或优化后的质量检查
- ✅ Bug 修复后的回归测试
- ✅ 多轮迭代中的稳定化阶段
- ✅ 准备发布前的最后检查

## 完整工作流

### 🔄 第一阶段：代码验证与基础测试

**目标：** 确保代码无语法错误且能正常导入

**步骤：**

1. **语法检查**
   ```bash
   python -m py_compile <file1.py> <file2.py> ...
   ```
   - 检查所有修改或新增的 Python 文件
   - ✅ 通过：无输出或输出成功信息
   - ❌ 失败：修复语法错误后重试

2. **模块导入测试**
   ```bash
   python -c "from module_name import Class; print('✓ Import successful')"
   ```
   - 测试核心模块的导入
   - 验证所有依赖是否可用
   - ✅ 通过：打印成功信息
   - ❌ 失败：检查缺失依赖或导入路径

3. **功能单元测试**
   ```bash
   python -m pytest tests/ -v  # 如果有测试文件
   # 或
   python <test_script.py>     # 手动测试脚本
   ```
   - 运行现有测试套件
   - 测试新增功能的核心逻辑
   - ✅ 通过：所有测试用例通过
   - ❌ 失败：分析失败原因并修复

**决策点：**
- 是否有测试框架？→ 使用 pytest
- 没有现成测试？→ 编写快速验证脚本

---

### 🧪 第二阶段：集成测试与功能验证

**目标：** 模拟真实用户操作，验证新功能端到端可用

**步骤：**

1. **环境准备**
   ```bash
   # 激活虚拟环境（如果需要）
   .\.venv\Scripts\Activate.ps1
   
   # 检查/安装必要的工具
   pip install -r requirements.txt
   ```
   - 确保开发环境干净
   - 安装所有必要依赖

2. **获取缺失工具**
   ```powershell
   # PowerShell 示例：检查并安装工具
   if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
       Write-Host "Python 未找到，正在安装..."
       # 安装命令
   }
   ```
   - 检查必要的命令行工具
   - 若缺失则通过 PowerShell 下载安装
   - 常见工具：Python、Git、Node.js 等

3. **功能场景测试**
   
   **场景 A：正常流程**
   ```bash
   # 测试核心功能
   python ai_auto_ps.py
   # 验证应用启动、UI 加载、功能可用
   ```
   - 模拟标准用户操作
   - 验证主要功能路径
   - 检查输出结果的正确性

   **场景 B：边界条件**
   ```bash
   # 测试异常输入处理
   python -c "
   from module import function
   try:
       result = function(invalid_input)
       print('✓ 异常处理正确')
   except Exception as e:
       print(f'✗ 错误: {e}')
   "
   ```
   - 测试边界值输入
   - 验证错误处理机制
   - 确保不会崩溃

   **场景 C：集成测试**
   ```bash
   # 测试多模块协作
   python -c "
   from module1 import ComponentA
   from module2 import ComponentB
   
   # 测试组件间交互
   result = ComponentA.process(ComponentB.generate())
   print(f'✓ 集成测试通过: {result}')
   "
   ```
   - 验证新功能与现有系统的配合
   - 检查数据流通过程
   - 确保没有隐藏的依赖问题

4. **性能/资源检查**（如适用）
   ```bash
   # 检查内存占用、执行时间等
   time python ai_auto_ps.py
   ```
   - 新功能是否有显著性能影响
   - 是否有内存泄漏风险
   - 执行时间是否在接受范围内

**决策点：**
- 测试是否全部通过？
  - ✅ 是 → 进入第三阶段
  - ❌ 否 → 分析失败原因，修复代码，重新测试

---

### 📝 第三阶段：文档更新

**目标：** 确保文档与代码保持同步

**步骤：**

1. **识别需要更新的文档**
   - README.md - 功能列表、使用方式
   - CHANGELOG.md - 更新日志（如有）
   - API 文档 - 新增函数/类的说明
   - 配置文档 - 新增环境变量、参数说明
   - UPGRADE_*.md - 升级指南

2. **更新 README.md**
   ```markdown
   ## 新增功能部分
   
   - **功能名称** - 简短描述
   - 主要特性列表
   - 使用示例（代码片段）
   - 配置说明
   
   ## 安装/更新说明
   
   - 新增依赖列表
   - 升级步骤
   ```
   
   检查清单：
   - ✅ 功能描述清晰准确
   - ✅ 包含使用示例
   - ✅ 新增依赖已列出
   - ✅ 配置说明完整
   - ✅ 目录结构最新

3. **更新变更日志**（如有 CHANGELOG.md）
   ```markdown
   ## [版本号] - YYYY-MM-DD
   
   ### Added
   - 新增功能 A：描述
   - 新增功能 B：描述
   
   ### Changed
   - 修改 X：说明变化
   
   ### Fixed
   - 修复 Bug Y：问题描述
   ```

4. **文档质量检查**
   ```bash
   # 验证 Markdown 格式
   python -c "
   import re
   with open('README.md') as f:
       content = f.read()
       links = re.findall(r'\[.*?\]\(.*?\)', content)
       print(f'✓ 找到 {len(links)} 个链接')
       # 检查链接有效性（可选）
   "
   ```
   - 检查 Markdown 格式正确
   - 验证链接有效性（特别是内部链接）
   - 确保代码示例语法高亮正确

**决策点：**
- 文档是否完整准确？
  - ✅ 是 → 进入第四阶段
  - ❌ 否 → 补充或修正文档，重新检查

---

### 🧹 第四阶段：工程文件清理

**目标：** 删除开发过程中的临时文件和无关文件，保持工程目录干净，仅保留核心程序文件和 SKILL.md

**步骤：**

1. **识别需要删除的文件**
   ```powershell
   # 查看当前目录结构（Windows）
   Get-ChildItem -Force

   # 查看 Git 未追踪的文件（重要：只有这些才可以考虑删除）
   git status --short

   # 识别以下无关文件（可根据实际情况调整）：
   # - Python 字节码缓存（__pycache__、*.pyc）
   # - /tmp 目录下的临时调试脚本
   ```

2. **制定清理清单**
   ```
   保留的文件/目录（仓库核心文件，绝对不可删除）：
   ✅ ai_auto_ps.py               # 核心程序
   ✅ multi_solution_generator.py # 核心功能模块
   ✅ solution_manager.py         # 版本管理模块
   ✅ requirements.txt            # 依赖文件
   ✅ README.md                   # 项目说明
   ✅ SKILL.md                    # 工作流文档（保留）
   ✅ tests/                      # 测试目录
   ✅ start.bat                   # Windows 启动脚本
   ✅ LICENSE                     # 许可证文件
   ✅ .gitignore                  # Git 忽略规则

   ⚠️  清理原则：
   - 仅删除 /tmp 或系统临时目录下由本次开发产生的临时文件
   - 绝对不要删除仓库中已被 Git 追踪的文件（可用 git status 确认）
   - 如果不确定某个文件是否应该保留，保留它，不要删除

   可安全清理的对象（仅限未被 Git 追踪的临时产物）：
   ❌ __pycache__/                # Python 字节码缓存（已在 .gitignore 中）
   ❌ *.pyc                       # 编译文件（已在 .gitignore 中）
   ❌ /tmp/ 下的调试脚本           # 只删 /tmp 目录内的临时文件
   ```

3. **执行清理（仅清理未追踪的临时产物）**
   ```powershell
   # 先确认文件是否已被 Git 追踪，已追踪的文件不可随意删除
   git status

   # 仅删除未追踪的 Python 缓存目录（安全）
   if (Test-Path "__pycache__") { Remove-Item "__pycache__" -Recurse -Force }

   # 删除 /tmp 下本次产生的临时脚本（按需）
   # Remove-Item "$env:TEMP\debug_temp.py" -ErrorAction SilentlyContinue
   ```

   ⚠️ **禁止操作：** 不要执行 `Remove-Item` 删除仓库根目录下任何 `.md`、`.py`、`.bat`、`.txt` 文件，这些均为受版本控制的文件。

4. **验证清理结果**
   ```powershell
   # 查看最终目录结构
   Get-ChildItem -Force

   # 确认关键文件存在
   Test-Path ai_auto_ps.py
   Test-Path SKILL.md
   Test-Path README.md
   Test-Path requirements.txt
   Test-Path start.bat
   ```

5. **提交清理结果**
   ```powershell
   git add .
   git commit -m "chore: 清理工程文件，删除临时产物

   - 删除 __pycache__ 等字节码缓存
   - 保留所有核心程序文件和文档
   "
   ```

   > ⚠️ **Agent 环境注意：** 不要执行 `git push`，直接调用 `report_progress` 工具完成提交推送。

**清理检查清单：**
- [ ] 已用 `git status` 确认只删除了未追踪的临时文件
- [ ] 核心程序文件完整（ai_auto_ps.py、solution_manager.py 等）
- [ ] SKILL.md 保留（工作流文档）
- [ ] README.md 保留（项目说明）
- [ ] Git 提交完成
- [ ] 目录结构清晰

**决策点：**
- 是否需要保留其他文档文件？
  - ✅ 是 → 有选择地保留必要的文档
  - 不确定？→ **保留**，不要冒险删除

---

### 🚀 第五阶段：版本控制与上传

**目标：** 将经过验证的版本通过平台工具推送到 GitHub

> ⚠️ **重要：Agent 环境限制**
> 在 GitHub Copilot Agent / 受限 CI 沙箱中，**直接执行 `git push` 命令会被系统拦截**。
> 必须使用 **`report_progress` 工具** 来完成提交与推送操作。
> 不要反复重试 `git push`，遇到拒绝时应立即切换到 `report_progress`。

**步骤：**

1. **检查 Git 状态**
   ```powershell
   git status
   git --no-pager diff --stat    # 查看改动摘要
   ```
   - 确认所有修改都已保存
   - 确认新文件已被追踪
   - 理解本次提交的所有改动

2. **阶段化提交（本地）**
   ```powershell
   # 方案 A：分类提交（推荐）
   git add ai_auto_ps.py multi_solution_generator.py
   git commit -m "feat: 添加多版本管理功能"

   git add README.md
   git commit -m "docs: 更新文档说明新功能"

   # 方案 B：一次性提交
   git add .
   git commit -m "feat: 完整迭代 - 多版本反馈系统

   - 新增 solution_manager.py 模块
   - 扩展 multi_solution_generator.py 支持理由
   - 升级 UI 支持多版本对比和反馈
   - 更新完整文档
   "
   ```

   提交信息规范：
   - `feat:` 新功能
   - `fix:` Bug 修复
   - `docs:` 文档更新
   - `refactor:` 代码重构
   - `test:` 测试相关
   - `perf:` 性能优化

3. **本地验证**
   ```powershell
   # 查看本地提交列表
   git --no-pager log --oneline -5

   # 验证当前分支
   git branch
   ```
   - 确认提交历史清晰
   - 确认在正确的分支
   - 确认没有本地未提交的改动

4. **通过 report_progress 工具推送到远程**

   > **Agent 环境专用步骤（替代 git push）**
   >
   > 调用 `report_progress` 工具，填写 `commitMessage` 与 `prDescription`。
   > 该工具会自动执行 `git add .`、`git commit`、`git push` 并更新 PR 描述。
   > **不要** 在终端手动执行 `git push origin <branch>`，该命令在 Agent 环境中被禁止。

   如需在 **本机 Windows 开发环境**（非 Agent 沙箱）中推送：
   ```powershell
   git push origin <当前分支名>
   # 例如：git push origin main
   ```
   - 📡 等待推送完成
   - ⏳ 若网络超时，稍后重试，不阻断工程流程

5. **远程验证（可选）**
   ```powershell
   # 推送完成后可在浏览器访问仓库确认：
   # 1. 查看最新提交是否已同步
   # 2. 验证文件内容正确
   # 3. 检查 GitHub Actions（如有 CI/CD）
   ```

**决策点：**
- 是否在 Agent 环境中运行？
  - ✅ 是 → **使用 report_progress 工具**，禁止手动 git push
  - ❌ 否（本机 Windows）→ 使用 `git push origin <branch-name>`
  - 遇到网络超时/拒绝 → **放弃当前推送尝试**，不要无限重试；本地 git commit 已安全保存

---

## 关键质量检查清单

在每个阶段完成前，检查以下项目：

### 代码质量
- [ ] 无语法错误
- [ ] 所有新模块可导入
- [ ] 依赖完整（在 requirements.txt 中）
- [ ] 代码风格一致

### 功能正确性
- [ ] 核心功能正常工作
- [ ] 边界条件处理正确
- [ ] 错误信息清晰有用
- [ ] 性能在可接受范围内

### 文档完整性
- [ ] 新功能已记录
- [ ] 使用示例清晰
- [ ] API 说明准确
- [ ] 配置指南完整
- [ ] 链接全部有效

### 版本管理
- [ ] 提交信息清晰准确
- [ ] 本地 git commit 已完成（必须）
- [ ] 通过 report_progress 工具或 git push 推送到远程（Agent 环境用前者；网络异常时可暂缓，不阻断工程）
- [ ] 分支状态正确

---

## 常见问题与解决方案

### Q: 测试失败了怎么办？
**A:** 
1. 查看错误信息确定问题所在
2. 修复代码
3. 重新运行测试（从失败的测试开始）
4. 确保所有测试通过后再进行下一步

### Q: 文档更新太耗时？
**A:**
- 优先更新 README.md 的功能列表和使用说明
- 可以使用代码中的 docstring 和注释自动生成部分文档
- 建立文档模板加快更新速度

### Q: 推送失败怎么办？
**A:**

**情况 A：在 Agent / CI 沙箱中运行**
- 直接执行 `git push` 在 Agent 环境中会被系统拦截，这是正常现象
- **解决方案：** 使用 `report_progress` 工具代替 `git push`，由平台完成推送
- 不要反复重试 `git push`，会持续失败

**情况 B：在本机 Windows 环境中运行**
```powershell
# 1. 检查网络连接
ping github.com

# 2. 检查认证信息
git config user.name
git config user.email

# 3. 查看具体错误
git --no-pager push --verbose origin <branch-name>

# 4. 若认证失败，更新凭据
git credential reject https://github.com
```

### Q: 如何只测试新增功能而不影响现有功能？
**A:**
- 为新功能编写独立的测试模块
- 使用功能开关/环境变量控制新功能启用
- 先测试新功能独立部分，再测试集成部分

---

## 变体与扩展

### 快速版（用于小改动）
1. 语法检查
2. 快速功能测试
3. 更新 README（仅相关部分）
4. 本地 git commit + 通过 report_progress 工具推送

**用时：** 5-15 分钟

### 完整版（用于大功能）
1. 完整代码验证
2. 详细集成测试
3. 完整文档更新
4. 详细提交消息
5. 创建 Release Note

**用时：** 30-60 分钟

### 发布版（准备版本发布）
1. 完整的所有测试
2. 性能基准测试
3. 文档和示例完全检查
4. 创建 Release 标签
5. 生成 CHANGELOG

**用时：** 1-2 小时

---

## 工具推荐

| 工具 | 用途 | 安装 |
|------|------|------|
| `pytest` | 单元测试框架 | `pip install pytest` |
| `flake8` | 代码风格检查 | `pip install flake8` |
| `black` | 代码格式化 | `pip install black` |
| `mypy` | 类型检查 | `pip install mypy` |
| `markdownlint` | Markdown 检查 | `npm install -g markdownlint-cli` |

---

## 成功指标

工作流完成时应该满足：

✅ **代码质量**
- 无语法/导入错误
- 新功能所有测试通过
- 没有 regression bugs

✅ **文档同步**
- README 已更新
- API 文档准确
- 使用示例可运行

✅ **版本控制**
- 本地 git commit 已完成
- 通过 report_progress 工具（Agent 环境）或 git push（本机）完成推送
- 提交历史清晰

---

## 相关技能推荐

- **代码审查技能** - 在提交前进行自我审查
- **问题排查技能** - 当测试失败时的调试方法
- **CI/CD 设置** - 自动化部分测试和检查过程
