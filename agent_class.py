"""阶段5-第1课：Agent 类的用法演示（实现已抽到 agent.py）

运行：python agent_class.py
观察：同一个 Agent 实例连问三句，第 2、3 句能答出第 1 句里的信息
     —— 记忆不是模型自带的，是我们把 messages 一直攒着传给它的。
"""

from agent import Agent

if __name__ == "__main__":
    agent = Agent()  # 造一个机器人（整个会话只造一次，记忆跟着它走）

    print(agent.chat("我叫小明，今年 20 岁"))
    print(agent.chat("我叫什么名字？"))   # 考它记性
    print(agent.chat("我今年多大？"))     # 再考一次

    print(f"\n一共聊了 {agent.turn_count} 轮，上下文 {len(agent.messages)} 条消息")
