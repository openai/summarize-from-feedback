def test_generator(split):
    for n in range(10):
        yield dict(
            query=f"This is the {n}th context in the dataset {split} split",
            reference=f"{n}th summary in {split} split",
        )
