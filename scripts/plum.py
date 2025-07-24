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