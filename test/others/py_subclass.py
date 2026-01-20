
class BaseSubClass:
    def __init_subclass__(cls, msg="Hi", **kwargs):
        super().__init_subclass__(**kwargs)

        # 将子类的类成员 name 移除
        # 并重写 __init__ 将 name 设为子类的实例成员
        name = getattr(cls, "name")
        delattr(cls, "name")

        old_init = getattr(cls, "__init__", None)

        def new_init(self, *args, **kwargs):
            self.name = name
            if old_init: old_init(self, *args, **kwargs)

        cls.__init__ = new_init

        # 加入一个实例方法 greet
        cls.greet = lambda self: print(msg, name)
    

# msg 会作为 BaseSubClass 的 __init_subclass__ 参数
class Derived(BaseSubClass, msg="Hello, World!"):
    name:str = "Derived"

if __name__ == "__main__":
    o = Derived()
    print(o.name) # "Derived"
    o.greet() # "Hello, World! Derived"

    print(hasattr(Derived, "name")) # False

