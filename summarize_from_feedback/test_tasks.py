import pytest
import torch

from summarize_from_feedback import tasks
from summarize_from_feedback.query_response_model import PADDING_TOKEN
from summarize_from_feedback.utils.assertions import assert_eq


class TestEncoder:
    def encode(self, text):
        return [ord(x) for x in text]

    def decode(self, tokens):
        return "".join([chr(x) for x in tokens])


def test_process_query():
    h = tasks.TaskQueryHParams(length=10, format_str="<{query}>")
    encoder = TestEncoder()

    query = "123456789abc"
    assert_eq(
        tasks.process_query(dict(query=query), encoder=encoder, hparams=h, pad_sequence=[0, 0, 0]),
        dict(tokens=encoder.encode("<" + query[:8] + ">")),
    )

    with pytest.raises(KeyError):
        tasks.process_query(dict(x=query), encoder=encoder, hparams=h, pad_sequence=[0, 0, 0])

    query = "12345a7"
    with pytest.raises(AssertionError):
        tasks.process_query(query, encoder=encoder, hparams=h, pad_sequence=[0, 0, 0])
    h.pad_side = "left"
    assert_eq(
        tasks.process_query(query, encoder=encoder, hparams=h, pad_sequence=[0, 0, 0]),
        dict(tokens=[0] + encoder.encode("<" + query + ">")),
    )


def test_process_response():
    encoder = TestEncoder()

    def test_response(h, response, expected_processed, expected_decoded):
        response_encoder = tasks.ResponseEncoder(h, encoder)

        processed_tensor = response_encoder.process_responses(torch.LongTensor([response]))
        processed = processed_tensor.numpy()[0]
        assert_eq(processed, expected_processed)

        decoded = response_encoder.decode_response(processed)
        assert_eq(decoded, expected_decoded)
        assert_eq(processed, response_encoder.encode_response(expected_decoded))
        decoded = response_encoder.decode_responses(processed_tensor)
        assert_eq(decoded, [expected_decoded])

    test_response(
        tasks.TaskResponseHParams(length=10, truncate_token=None),
        encoder.encode("123456789a"),
        encoder.encode("123456789a"),
        "123456789a",
    )

    test_response(
        tasks.TaskResponseHParams(length=10, truncate_token=ord("5")),
        encoder.encode("123456789a"),
        encoder.encode("12345") + [PADDING_TOKEN] * 5,
        "1234",
    )


def test_encode_response():
    encoder = TestEncoder()

    def test_encode_response(h, response, expected_processed=None, allow_truncate=False):
        response_encoder = tasks.ResponseEncoder(h, encoder)
        processed = response_encoder.encode_response(response, allow_truncate=allow_truncate)
        if expected_processed is not None:
            assert_eq(expected_processed, processed)

    test_encode_response(
        tasks.TaskResponseHParams(length=10, truncate_token=None),
        "123456789a",
        encoder.encode("123456789a"),
    )

    test_encode_response(
        tasks.TaskResponseHParams(length=10, truncate_token=None),
        "123456789",
        encoder.encode("123456789") + [PADDING_TOKEN],
    )

    with pytest.raises(AssertionError):
        test_encode_response(
            tasks.TaskResponseHParams(length=10, truncate_token=None),
            "123456789ab",
            encoder.encode("123456789a"),
        )

    test_encode_response(
        tasks.TaskResponseHParams(length=10, truncate_token=None),
        "123456789ab",
        encoder.encode("123456789a"),
        allow_truncate=True,
    )

    test_encode_response(
        tasks.TaskResponseHParams(length=10, truncate_token=ord("5")),
        "1234",
        encoder.encode("12345") + [PADDING_TOKEN] * 5,
    )

    with pytest.raises(AssertionError):
        test_encode_response(
            tasks.TaskResponseHParams(length=10, truncate_token=ord("b")),
            "123456789a",
            encoder.encode("12345") + [PADDING_TOKEN] * 5,
        )

    test_encode_response(
        tasks.TaskResponseHParams(length=10, truncate_token=ord("b")),
        "123456789a",
        encoder.encode("123456789b"),
        allow_truncate=True,
    )

    test_encode_response(
        tasks.TaskResponseHParams(length=10, truncate_token=ord("b")),
        "123456789aaaab",
        encoder.encode("123456789b"),
        allow_truncate=True,
    )
