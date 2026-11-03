"""阶段5-第1课：Agent 类的用法演示（实现已抽到 agent.py）

运行：python agent_class.py
观察：同一个 Agent 实例连问三句，第 2、3 句能答出第 1 句里的信息
     —— 记忆不是模型自带的，是我们把 messages 一直攒着传给它的。

注意这里的 print 都写在 printer 里，Agent 类本身一行 print 都没有：
类只负责"发生了什么"，怎么展示交给调用方。
"""

from agent import Agent


def printer(event: dict) -> None:
    """事件回调：把 Agent 的内部动作翻译成人能看的进度。"""
    if event["type"] == "step":
        print(f"[第{event['step']}步] 思考中...")
    elif event["type"] == "tool_start":
        print(f"[第{event['step']}步] 调用 {event['tool']}({event['args']})")
    elif event["type"] == "tool_end":
        brief = event["result"].replace("\n", " ")[:80]
        print(f"           结果: {brief}")
    elif event["type"] == "max_steps":
        print("达到最大步数，强制停止（防失控）")


if __name__ == "__main__":
    agent = Agent(on_event=printer)  # 造一个机器人（整个会话只造一次，记忆跟着它走）

    print("\n--- 第1句 ---")
    print(agent.chat("我叫小明，今年 20 岁"))
    print("\n--- 第2句 ---")
    print(agent.chat("我叫什么名字？"))   # 考它记性
    print("\n--- 第3句 ---")
    print(agent.chat("我今年多大？"))     # 再考一次

    print(f"\n一共聊了 {agent.turn_count} 轮，上下文 {len(agent.messages)} 条消息")
