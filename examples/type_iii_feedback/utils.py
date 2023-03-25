import random


class DummyTrial:

    def suggest_int(self, name, _min, _max):
        return random.randint(_min, _max)

    def suggest_float(self, name, _min, _max):
        return random.uniform(_min, _max)

    def suggest_categorical(self, name, cats):
        return random.choice(cats)

    def set_user_attr(self, v0, v1):
        pass

