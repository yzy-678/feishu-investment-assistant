# Railway Runbook

This checklist keeps the Feishu investment assistant recoverable after code,
Railway, or Feishu configuration changes.

## Required Variables

Set these in Railway `Variables` for the production service:

| Variable | Expected value |
| --- | --- |
| `DEEPSEEK_API_KEY` | DeepSeek API key |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` |
| `DEEPSEEK_MODEL` | `deepseek-chat` |
| `FEISHU_APP_ID` | Feishu app ID, starts with `cli_` |
| `FEISHU_APP_SECRET` | Feishu app secret |
| `FEISHU_BOT_NAME` | Bot display name |
| `FEISHU_EVENT_VERIFY_TOKEN` | Same value as Feishu event verification token |
| `ADMIN_USER_OPEN_ID` | Admin user's Feishu open_id, starts with `ou_` |
| `LOG_LEVEL` | `INFO` for normal production, `DEBUG` only during diagnosis |
| `DATABASE_PATH` | `config/investment.db` |
| `DATA_SOURCE` | `eastmoney` |

Optional variables:

| Variable | When to use |
| --- | --- |
| `FEISHU_EVENT_ENCRYPT_KEY` | Only when Feishu event encryption is enabled |
| `API_BEARER_TOKEN` | Only when protected internal API calls are used |

## Feishu Callback

In Feishu Open Platform, keep these settings aligned with Railway:

| Setting | Value |
| --- | --- |
| Event request URL | `https://feishu-investment-assistant-production.up.railway.app/feishu/event` |
| Callback request URL | Same as event request URL |
| Verification token | Same as `FEISHU_EVENT_VERIFY_TOKEN` |
| Event subscription | `im.message.receive_v1` |
| Bot ability | Enabled |

If the verification token changes in Feishu, update Railway first, redeploy,
then save the Feishu callback settings again.

## Redeploy

Use this after every push to `main`:

1. Open Railway project.
2. Open service `feishu-investment-assistant`.
3. Open `Deployments`.
4. Confirm the latest Git commit is the commit you want to run.
5. Click `Redeploy`.
6. Wait until the deployment status is `Active`.

The current production branch is `main`.

## Smoke Test

After deployment, test these in Feishu:

| Message | Expected route |
| --- | --- |
| `你好` | GeneralAgent |
| `今天市场怎么样` | MarketAgent with EastMoney data |
| `分析 000001` | MarketAgent with EastMoney quote |
| `生成早报` | ReportAgent |
| `查看预警` | AlertAgent |
| `立即扫描` | AlertAgent manual scan |

## Log Checks

For normal production, keep `LOG_LEVEL=INFO`.

Search Railway logs for:

| Keyword | Meaning |
| --- | --- |
| `EastMoney request failed` | A data source endpoint failed and retried/fell back |
| `EastMoney transient status` | EastMoney returned `502/503/504`, retry is active |
| `HTTP/1.1 200 OK` from Feishu reply API | Bot reply was sent successfully |
| `DeepSeekError` | DeepSeek API problem, often balance or key related |

Use `LOG_LEVEL=DEBUG` temporarily only when diagnosing EastMoney request details.
Then Railway will also print request URL, headers, status, redirect history, and
the first 500 response characters.

## Current Stable State

- Feishu is the only user entry point.
- `DATA_SOURCE=eastmoney` is the production market data source.
- EastMoney quote requests prefer `push2delay.eastmoney.com`, with fallback to
  `push2.eastmoney.com`.
- EastMoney historical K-line requests try `push2his.eastmoney.com`,
  `81.push2his.eastmoney.com`, and `82.push2his.eastmoney.com`.
- DeepSeek is charged only when an agent calls chat completion.
