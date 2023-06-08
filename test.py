#! /usr/bin/python3

from main import equal, compare_classification
import torch
import configs

tsr1 = torch.rand(2, 6)
tsr_cpy = tsr1


def test1_eq():
    assert equal(tsr1, tsr_cpy) == True, " [-] expected to be same tensors"


def test2_eq():
    tsr_false = torch.clone(tsr1)
    tsr_false[1, 0] = 2
    assert equal(tsr_false, tsr_cpy) == False, " [-] expected to different tensors"


def test1_cmp():
    assert (
        compare_classification(
            torch.clone(tsr1),
            torch.clone(tsr_cpy),
            top_k=configs.TOP_K_MAX,
            logger=None,
        )
        == 0
    ), " [-] expected 0 errors"


def test2_cmp():
    tsr_false = torch.clone(tsr1)
    tsr_false[1, 0] = 2
    res: int = compare_classification(
        torch.clone(tsr_false),
        torch.clone(tsr_cpy),
        top_k=configs.TOP_K_MAX,
        logger=None,
    )
    assert res != 0, " [-] expected more than 0 errors"


if __name__ == "__main__":
    print(" [+] equal function tests")
    test1_eq()
    test2_eq()
    print(" [+] compare_classification function tests")
    test1_cmp()
    test2_cmp()
