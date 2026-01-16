from pinglab.multiprocessing import parallel


def test_parallel_executes_all():
    cfgs = [{"x": 1}, {"x": 2}, {"x": 3}]

    def inner(cfg):
        return cfg["x"] * 2

    results = parallel(inner, cfgs, label="test")
    assert sorted(results) == [2, 4, 6]
