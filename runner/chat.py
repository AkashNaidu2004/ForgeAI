"""Generic chat runner:  python -m runner.chat path/to/agent_definition.json"""
import sys
from forge.agent import Agent, load_definition


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python -m runner.chat <agent_definition.json>")
    d = load_definition(sys.argv[1])
    agent = Agent(d)
    print(f"[{d['name']}] {d.get('description', '')}\n(type 'exit' to quit)\n")
    while True:
        q = input("you> ").strip()
        if q.lower() in ("exit", "quit", ""):
            break
        r = agent.answer(q)
        tag = " [REFUSED]" if r["refused"] else ""
        print(f"agent{tag}> {r['answer']}")
        if r["sources"]:
            print(f"  sources: {', '.join(r['sources'])}")
        for t in r["tool_calls"]:
            print(f"  tool: {t['tool']}({t['args']}) -> {t['result']}")


if __name__ == "__main__":
    main()
