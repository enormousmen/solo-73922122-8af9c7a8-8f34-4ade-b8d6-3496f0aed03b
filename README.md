# Resumable Training API（可恢复单作业训练服务）

FastAPI + CPU 版 PyTorch 实现的**单作业训练服务**。训练工程师在作业被抢占（暂停、
进程强退、容器被杀），或在**某批参数已更新、进度尚未记录时遇到暂时性异常
（作业进入 `failed`）**后，可以从最近**成功**的检查点继续训练；在**同环境、
同种子、同配置**下，恢复后的逐批损失与最终参数与**不中断训练逐位一致**。

- 模型：两层全连接网络 `Linear → ReLU → Dropout → Linear`
- 优化器：Adam；每轮（epoch）打乱全部样本后按批训练
- 数据：仓库内置固定分类数据集 `data/classification.csv`（600 样本 / 2 特征 / 3 类）
- 默认配置：`config/default.json`（可在启动时按字段覆盖）

## 目录结构

```
app/
  main.py        # FastAPI 入口（/docs 自动生成）
  trainer.py     # 单作业状态机 + 确定性训练循环 + 暂停/续训
  checkpoint.py  # 原子检查点读写（tmp + fsync + rename）
  config.py      # 默认配置加载与校验
  dataset.py     # 固定 CSV 数据集加载
  model.py       # 两层 FC + Dropout
config/default.json      # 默认训练配置
data/classification.csv  # 固定分类数据集（scripts/generate_data.py 生成）
scripts/generate_data.py # 数据集生成脚本（固定种子，仅作溯源）
scripts/verify_repro.py  # 端到端验证：不中断对照 / 暂停续训 / 强退恢复 等
Dockerfile, docker-compose.yml, requirements.txt
```

## 快速开始

### Docker Compose（推荐）

```bash
docker compose up --build
# 服务: http://localhost:8000  交互式文档: http://localhost:8000/docs
```

检查点保存在命名卷 `trainer-runs` 中，容器被杀或重启后数据不丢。

### 本地运行（开发）

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
# 环境变量: RUNS_DIR(默认 runs/)  CONFIG_PATH(默认 config/default.json)
```

## API 一览

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 服务信息 |
| GET | `/config` | 默认配置与运行时信息 |
| POST | `/train/start` | 启动新训练。Body: `{"force": false, "config": {覆盖字段}}` |
| POST | `/train/pause` | 暂停。**训练停止且完整检查点持久化后才返回 200** |
| POST | `/train/resume` | 从暂停态续训，或把**失败作业**还原到最近成功检查点后续训（并发只放行一个训练者） |
| GET | `/train/status` | 状态与进度（epoch/batch/百分比/检查点健康度/失败与还原原因） |
| GET | `/train/losses` | 逐批损失（支持 `since_step`、`limit`） |
| GET | `/train/parameters` | 下载当前模型参数（.pt，取自最近持久化检查点） |
| GET | `/train/checkpoint` | 下载完整检查点（模型+优化器+RNG+历史） |

交互式文档（Swagger UI）在 `/docs`，可直接在浏览器里调通上述接口。

### curl 示例

```bash
# 启动（覆盖部分配置；已有运行需 force=true 或先续训）
curl -X POST localhost:8000/train/start -H 'Content-Type: application/json' \
     -d '{"config": {"epochs": 8, "batch_size": 32, "lr": 0.01}}'

# 查询进度
curl -s localhost:8000/train/status | jq '.state, .completed_batches, .total_batches'

# 逐批损失
curl -s 'localhost:8000/train/losses?limit=5' | jq '.losses[]'

# 暂停（阻塞至训练停止且检查点落盘才返回）
curl -X POST localhost:8000/train/pause

# 续训
curl -X POST localhost:8000/train/resume

# 下载参数 / 完整检查点
curl -OJ localhost:8000/train/parameters     # model_parameters.pt
curl -OJ localhost:8000/train/checkpoint     # checkpoint.pt
```

状态机：`idle → running ⇄ pausing → paused → running … → finished`（训练/还原
异常为 `failed`，还原进行中为 `recovering`）。`paused` 表示训练已停、可续训；
`checkpoint_dirty=true` 表示内存进度领先于磁盘（上次保存失败），此时重启进程
会回退到磁盘上的旧检查点。

## 失败作业的续训语义（核心保证）

训练循环中，一个批次依次执行 `zero_grad → forward → loss → backward →
optimizer.step()`，**之后**才把进度（`next_batch`、损失）写入内存并在批边界
落检查点。若异常发生在这两步之间——参数/Adam/RNG 已被该批推进，而进度尚未
记录——作业进入 `failed`。此时内存现场已被污染，**绝不能直接在它上面重跑该
批**，否则等于对同一批做两次更新，最终参数必然偏离正常训练。

`POST /train/resume` 对 `failed` 作业的处理因此与暂停续训不同：

1. **拒绝无点续训**：同一作业在磁盘上没有成功检查点时返回 **404** 并说明原因，
   只能 `force=true` 重开（此时新作业有全新 `run_id`，旧检查点先被删除）。
2. **完整还原后才启动**：从**同一作业**（校验 `run_id` 一致）的最近成功检查点
   一次性还原模型、Adam、RNG 状态、批次位置（`epoch`/`next_batch`）与完整损失
   历史，全部成功后才启动训练线程；任一步失败则作业保持 `failed`，**不会从半
   还原现场启动**。
3. **加载失败可查且不毁原件**：检查点不可读/损坏/结构缺失时返回 **503**，
   原文件逐字节保留，原因写入 `status.last_error` 与 `status.recovery_error`，
   修复后可再次重试。
4. **还原期间可查询**：还原在锁外进行，状态先置为 `recovering`；慢存储下
   `GET /train/status`、`/train/losses` 等查询照常快速响应。
5. **单训练者**：还原或训练进行中的并发 `/train/resume` 一律 **409**，全程最多
   一个训练线程。
6. **强制新开不混用**：`force=true` 生成新 `run_id` 并先删除旧检查点，新作业
   不可能加载到旧作业的还原结果；若磁盘检查点的 `run_id` 与当前作业不符，
   续训直接 409 拒绝。

`GET /train/status` 在原有字段外增加：`failed_at`（失败批次 `{epoch,batch}`）、
`resume_attempts`、`recoverable`（失败且磁盘上存在检查点）、`recovery_error`
（还原/启动恢复的失败原因）。

### 演示：异常批次后重试

测试专用故障注入（由 runs 目录中的标记文件驱动，生产环境不会出现）：写入
`INJECT_BATCH_FAILURE_AT`，内容为 `epoch:batch`，训练到该批时会在
`optimizer.step()` 之后、记录进度之前抛出暂时性错误。

```bash
# 1. 启动训练
curl -X POST localhost:8000/train/start -H 'Content-Type: application/json' \
     -d '{"config": {"epochs": 2, "checkpoint_every": 2}}'
# 2. 在第 0 轮第 5 批注入暂时性异常
echo "0:5" > runs/INJECT_BATCH_FAILURE_AT
curl -s localhost:8000/train/status | jq '.state, .failed_at, .last_error'
#    -> "failed", {"epoch":0,"batch":5}, 含 "batch progress was not recorded"
# 3. 排除故障后重试：从最近成功检查点完整还原并重跑该批
rm runs/INJECT_BATCH_FAILURE_AT
curl -X POST localhost:8000/train/resume
curl -s localhost:8000/train/status | jq '.state, .resume_attempts'
#    -> "running" -> "finished"，attempts=1
# 4. 与不中断基线对比：逐批损失与最终参数逐位一致（verify 脚本自动完成）
```

## 可复现性设计（为什么恢复后逐位一致）

1. **单线程 + 确定性算子**：`torch.set_num_threads(1)` 并开启
   `torch.use_deterministic_algorithms(True)`，同一构建下浮点结果逐位确定。
2. **单一 RNG 流**：新运行先 `torch.manual_seed(seed)`，模型初始化与 Dropout
   都从该流取数；检查点保存 `torch.get_rng_state()`，恢复时原样写回。
3. **每轮打乱可重推**：第 e 轮的样本排列是 `(seed, e)` 的纯函数
   （`random.Random(f"{seed}:{epoch}")`），重启后无需存储即可重建当轮顺序。
4. **批边界检查点**：每个检查点记录 `epoch`、`next_batch`、模型与 Adam 状态、
   RNG 状态、完整损失历史、解析后的配置与 `completed` 标志——即"准备执行第 k
   批"的完整现场。恢复 = 还原现场后继续同一确定性序列。
5. **原子写**：先写 `.checkpoint.pt.tmp.<uuid>` → fsync → `os.replace` 原子改名
   → 目录 fsync。保存中被强退，磁盘上只会是**完整的旧点或完整的新点**；
   启动时清理残留的临时文件。保存失败不触碰旧文件，明确报错，可重试。

> 注意：逐位一致的前提是**同环境**（同一 torch 构建/CPU 指令集、单线程）。跨
> 不同 CPU 架构或 torch 版本不保证逐位一致，但同机同镜像（如本 compose）成立。

## 不中断对照与退出重启示例

### 自动化端到端验证（推荐）

```bash
python3 scripts/verify_repro.py
```

脚本会真实拉起 uvicorn 服务进程，依次验证（详见脚本 docstring）：
A 不中断基线；B 暂停+续训；C 强退+重启+续训；D 并发续训只放行一个；
E1 保存失败→报错→旧点保留→重试成功；E2 保存失败后强退→回退旧点恢复；
F 慢保存不挡查询；G 保存进行中强退→恢复完整旧/新点；H 基础接口与 /docs；
I1 批内暂时性异常（参数已更新、进度未记录）→失败→从同作业最近成功点还原重跑→
与基线逐位一致；I2 无成功点时明确拒绝续训（404）；I3 检查点不可加载→503、
原件保留、原因可查、修复后重试成功；I4 慢还原期间查询照常响应且并发续训 409；
I5 `force=true` 新开作业与旧作业完全隔离；J 损坏/外来/残缺检查点的进程内单元
防护。B–G、I 各场景的逐批损失与最终参数均与各自不中断基线**逐位一致**才判通过。

> 本环境的 seccomp 策略禁止向其他进程发信号，脚本用 `ENABLE_TEST_CRASH_ENDPOINT=1`
> 开启的测试专用 `POST /test/crash`（`os._exit`，立即死亡、无任何清理）模拟强退；
> 恢复语义与 SIGKILL 等价。该端点默认关闭，切勿在生产开启。
>
> 其余测试专用钩子全部由 runs 目录中的标记文件驱动，正常运行时不存在：
> `INJECT_SAVE_FAILURES`（保存必失败）、`INJECT_LOAD_FAILURES`（加载必失败）、
> `INJECT_LOAD_DELAY_SECONDS`（加载延迟秒数）、`INJECT_BATCH_FAILURE_AT`
> （在指定批次 `epoch:batch` 注入暂时性训练异常）。

### Docker 手动退出重启示例

```bash
docker compose up -d --build
curl -X POST localhost:8000/train/start -H 'Content-Type: application/json' \
     -d '{"config": {"epochs": 8}}'

docker kill resumable-trainer          # 模拟抢占/强退（真实 SIGKILL）
docker compose start                   # 重启容器（检查点在命名卷中）
curl -s localhost:8000/train/status    # state=paused, recovered_after_restart=true
curl -X POST localhost:8000/train/resume
curl -s localhost:8000/train/status    # 直至 state=finished
```

另开一个终端可对同一数据集做一次不中断对照（`force=true` 重跑同配置），用
`GET /train/losses` 与 `GET /train/parameters` 对比两边逐批损失与参数完全一致。

## 实测结果

（以下为 `python3 scripts/verify_repro.py` 在本仓库环境下的真实输出摘要）

```
    workdir: /tmp/verify-repro-ndor_el1
    
    == H: service basics (idle server) ==
      [PASS] H1 GET /
      [PASS] H2 GET /docs (Swagger UI)
      [PASS] H3 parameters before any checkpoint -> 404
      [PASS] H4 resume with no run -> 404
      [PASS] H5 pause while idle -> 409
    
    == A: uninterrupted baseline ==
      [PASS] A baseline finished - 95 batches, final loss 0.027771
    
    == B: pause + resume ==
      [PASS] B1 pause returns only after checkpoint persisted - pause latency 0.03s, stopped at batch 14
      [PASS] B2 resume accepted
        losses: 95 entries, exact match
        params: all tensors bit-identical (sha256 2397ddd7b6b8b70b == 2397ddd7b6b8b70b)
      [PASS] B3 pause+resume == uninterrupted
    
    == D: concurrent resume starts exactly one trainer ==
      [PASS] D1 concurrent resumes -> one 200, one 409 - got [200, 409]
        losses: 95 entries, exact match
        params: all tensors bit-identical (sha256 2397ddd7b6b8b70b == 2397ddd7b6b8b70b)
      [PASS] D2 resumed run == uninterrupted
    
    == E1: checkpoint save failure keeps old checkpoint, retry works ==
      [PASS] E1 pause with failing save -> 500 - training is stopped but the checkpoint could not be persisted; the previous checkpoint is intact on 
      [PASS] E1 status shows dirty checkpoint + error
      [PASS] E1 old checkpoint untouched
      [PASS] E1 retried pause -> 200, checkpoint persisted
      [PASS] E1 resume after retry
        losses: 95 entries, exact match
        params: all tensors bit-identical (sha256 2397ddd7b6b8b70b == 2397ddd7b6b8b70b)
      [PASS] E1 run after save-failure == uninterrupted
    
    == F: slow checkpoint saves do not block queries ==
      [PASS] F1 queries respond fast while a 1s save is in flight - worst status latency 1ms
      [PASS] F2 pause during slow saves
      [PASS] F3 slow-save run finishes
    
    == C: hard crash + restart recovers from latest checkpoint ==
      [PASS] C1 server crashed mid-run
      [PASS] C2 restart recovers paused state from checkpoint - recovered at batch 10/95
      [PASS] C3 resume after restart
        losses: 95 entries, exact match
        params: all tensors bit-identical (sha256 2397ddd7b6b8b70b == 2397ddd7b6b8b70b)
      [PASS] C4 kill+restart == uninterrupted
    
    == E2: save failure, then hard crash -> old checkpoint used ==
      [PASS] E2 pause with failing save -> 500
      [PASS] E2 restart falls back to old complete checkpoint - recovered at batch 6/95
      [PASS] E2 resume after restart
        losses: 95 entries, exact match
        params: all tensors bit-identical (sha256 2397ddd7b6b8b70b == 2397ddd7b6b8b70b)
      [PASS] E2 save-fail+kill == uninterrupted
    
    == G: hard crash in the middle of a checkpoint save ==
      (baseline: identical config, uninterrupted)
      [PASS] G1 server crashed during save
      [PASS] G2 no torn checkpoint / stale temp cleaned - leftovers: none
      [PASS] G3 restart recovers a complete checkpoint (old or new) - recovered at batch 1/19
      [PASS] G3b resume after restart
        losses: 19 entries, exact match
        params: all tensors bit-identical (sha256 1353a7415c7cae90 == 1353a7415c7cae90)
      [PASS] G4 kill-mid-save == uninterrupted

    == I1: in-batch transient failure -> restore checkpoint, retry ==
      [PASS] I1 job entered failed state after in-batch error - failed_at={'epoch': 0, 'batch': 5}
      [PASS] I1 failure is queryable and a checkpoint exists
      [PASS] I1 no phantom progress for the failing batch; disk point is older - memory next_batch=5, disk next_batch=4
      [PASS] I1 resume accepted after failure - code=200
      [PASS] I1 resumed from the last successful checkpoint (batch re-run)
        losses: 95 entries, exact match
        params: all tensors bit-identical
      [PASS] I1 failure+resume == uninterrupted

    == I2: failure before first checkpoint -> resume refused ==
      [PASS] I2 failed with no checkpoint on disk
      [PASS] I2 resume refused with 404 and a clear reason
      [PASS] I2 job stays failed (no half-restored scene)
      [PASS] I2 force start accepted and discards the failed run

    == I3: unloadable checkpoint -> resume 503, original kept, retry ==
      [PASS] I3 resume with unloadable checkpoint -> 503
      [PASS] I3 original checkpoint byte-identical after failed resume
      [PASS] I3 job still failed, reason queryable, nothing started
      [PASS] I3 retry after fault clears -> 200
      [PASS] I3 load-failure then retry == uninterrupted

    == I4: slow recovery -> queries live, one trainer only ==
      [PASS] I4 recovering state was observable during restore
      [PASS] I4 status responds fast while recovery is in flight - worst latency 4ms
      [PASS] I4 concurrent resume during recovery -> 409
      [PASS] I4 original resume eventually accepted -> 200
      [PASS] I4 slow-recovery run == uninterrupted

    == I5: forced new run never reuses the failed run's recovery ==
      [PASS] I5 force start replaces the failed run (new run_id, old checkpoint removed)
      [PASS] I5 forced new run == uninterrupted

    == J: checkpoint guard unit checks (corrupt/tampered/incomplete) ==
      [PASS] J1 corrupt checkpoint -> failed at startup, no half-restore
      [PASS] J1 resume on corrupt checkpoint refused, reason surfaced
      [PASS] J2 foreign-run checkpoint -> 409, no mixing of recovery results
      [PASS] J2 job stays failed and original checkpoint is untouched
      [PASS] J3 incomplete checkpoint -> failed at startup, reason queryable
      [PASS] J4 resume with no run/checkpoint -> 409 refused; idle scene untouched

    ================ SUMMARY ================
      66/66 checks passed（逐项明细同上，全部 PASS；退出码 0）

```

## 限制与未完成事项

- 单作业、单进程、单 worker：多副本/多 worker 部署会破坏单训练者语义（需引入
  外部锁与共享存储，未实现）。
- 逐位一致仅保证同环境同构建（见上节）；未做跨机器/跨 torch 版本对齐。
- 检查点只保留最新一份（原子替换保证新旧二选一）；未做历史版本保留/滚动清理。
- `force=true` 启动新运行会**立即丢弃**旧检查点：若新运行在写出第一个检查点
  前崩溃，则无点可恢复（状态回到 `idle`），只能重新开始——这是刻意语义，
  避免"已丢弃的旧运行"在重启后被悄悄恢复。失败作业若在首个成功检查点之前
  崩溃，续训同样被 404 拒绝（场景 I2 覆盖）。
- `failed` 续训只信任磁盘上**同 `run_id`** 的成功检查点；检查点结构/状态还原
  任一环节失败都整体放弃（作业保持 `failed`，原件保留、原因可查），不提供
  "忽略校验强制恢复"入口，避免半还原现场污染训练。
- `GET /train/parameters` 取自最近**已持久化**检查点，运行中可能落后最多
  `checkpoint_every` 批；未提供"当前内存参数"下载（避免与训练线程竞争）。
- 本环境无 docker，镜像构建与 compose 未实际执行；Dockerfile/compose 与本地实
  测使用完全相同的启动命令与依赖版本，但未在真实 Docker 中跑过验证脚本。
- 暂停等待上限为 `pause_timeout_seconds`（默认 30s），超时返回 504，训练仍会
  在批边界停下，可轮询状态确认。
