from plum import dispatch

class A:
    @dispatch
    def __init__(self) -> None:
        self.x = 0

    @dispatch
    def __init__(self, x: int) -> None:
        self.x = x

## init method also works
assert A().x == 0
assert A(1).x == 1

### async also works

import asyncio

@dispatch
async def count(n):
    print("starting count")
    for i in range(n):
        await asyncio.sleep(1)
        print(i)

@dispatch
async def count():
    await count(3)
