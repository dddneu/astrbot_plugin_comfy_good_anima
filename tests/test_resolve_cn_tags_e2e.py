"""端到端 async 流程测试（mock LLM）"""
import sys
sys.path.insert(0, '.')

import asyncio
from anima_agent.tag_service._ner import _parse
from anima_agent.tag_service._retrieval import get_engine
from anima_agent.tag_service.cn_tag_resolver import resolve_cn_tags


# Mock LLM: 直接返回 NER JSON
MOCK_JSON = '{"characters":[{"name":"可萝尔","context_series":"坎公骑冠剑","aliases":[]},{"name":"爱丽丝","context_series":"东方Project","aliases":[]}],"negative_elements":[]}'


async def mock_llm(sys_p, user_p):
    return MOCK_JSON


async def main():
    confirmed, nltags, negative = await resolve_cn_tags("画一个可萝尔和爱丽丝", mock_llm)
    print(f"confirmed: {confirmed}")
    print(f"nltags: {nltags}")
    print(f"negative: {negative}")
    assert confirmed, "应该有确认的 tag"
    print("PASS: 端到端流程 OK")


if __name__ == "__main__":
    asyncio.run(main())
