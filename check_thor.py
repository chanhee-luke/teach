from ai2thor.controller import Controller

c = Controller()
event = c.step(dict(action="MoveAhead"))
assert event.frame.shape == (300, 300, 3)
print(event.frame.shape)
print("Everything works!!!")
