"""ChainSink against the real Attendance contract on an in-memory EVM (eth-tester), plus outbox integration."""
import json
from pathlib import Path

import pytest
from conftest import captured

pytest.importorskip("eth_tester")
from web3 import EthereumTesterProvider, Web3  # noqa: E402

from app.chain import ChainSink, load_contract_address  # noqa: E402
from app.outbox import Outbox  # noqa: E402
from app.sinks import DeliveryRejected, DeliveryRetry  # noqa: E402
from app.worker import Stats, on_delivered  # noqa: E402

CONTRACT = json.loads((Path(__file__).parent / "fixtures" / "attendance_contract.json").read_text())


@pytest.fixture
def w3():
    return Web3(EthereumTesterProvider())


@pytest.fixture
def deployed(w3):
    c = w3.eth.contract(abi=CONTRACT["abi"], bytecode=CONTRACT["bin"])
    receipt = w3.eth.wait_for_transaction_receipt(c.constructor().transact({"from": w3.eth.accounts[0]}))
    return w3.eth.contract(address=receipt.contractAddress, abi=CONTRACT["abi"])


def records(contract):
    return [contract.functions.getRecord(i).call()[1] for i in range(contract.functions.getRecordCount().call())]


def event(i, member_id="202400004"):
    return {"member_id": member_id, "confidence": 0.7, "camera_id": "door-1", "captured_at": captured(i).isoformat(),
            "tx_hash": "", "block_number": None, "event_id": f"e{i}"}


def test_writes_member_to_contract(w3, deployed):
    sink = ChainSink(w3, deployed.address)
    assert sink.check()["owner"] == sink.sender
    result = sink.deliver(event(0))
    assert result["status"] == "recorded" and result["tx_hash"].startswith("0x") and result["block_number"] >= 1
    assert records(deployed) == ["202400004"]


def test_signs_with_private_key(w3, deployed):
    # eth-tester's first account; its key is public test data
    key = w3.provider.ethereum_tester.backend.account_keys[0]
    sink = ChainSink(w3, deployed.address, private_key=key.to_hex() if hasattr(key, "to_hex") else key)
    sink.deliver(event(0))
    assert records(deployed) == ["202400004"]


def test_non_owner_is_retried_not_recorded(w3, deployed):
    sink = ChainSink(w3, deployed.address, sender=w3.eth.accounts[1])
    with pytest.raises(DeliveryRetry):
        sink.deliver(event(0))
    assert records(deployed) == []


def test_imposter_is_rejected(w3, deployed):
    with pytest.raises(DeliveryRejected):
        ChainSink(w3, deployed.address).deliver(event(0, member_id=None))


def test_resume_does_not_write_twice(w3, deployed):
    sink = ChainSink(w3, deployed.address)
    saved = []
    first = sink.deliver(event(0), checkpoint=saved.append)
    assert saved == [first["tx_hash"]]
    # crash after sending, before the outbox row was deleted: the retry finds the mined tx
    assert sink.deliver(event(0), progress=saved[0]) == first
    assert records(deployed) == ["202400004"]


def test_outbox_to_chain_in_order(w3, deployed, tmp_path):
    stats = Stats()
    box = Outbox(str(tmp_path / "o.sqlite3"), ChainSink(w3, deployed.address).deliver, on_delivered=on_delivered(stats))
    for i, mid in enumerate(["A", "B", "C"]):
        box.put(event(i, member_id=mid))
    assert box.deliver_due() == 3
    assert records(deployed) == ["A", "B", "C"]
    assert stats.counters["recorded"] == 3 and box.counts() == {"pending": 0, "dead": 0}


def test_node_down_keeps_events_queued(tmp_path):
    w3 = Web3(Web3.HTTPProvider("http://127.0.0.1:9", request_kwargs={"timeout": 1}))
    box = Outbox(str(tmp_path / "o.sqlite3"), ChainSink(w3, "0x" + "11" * 20, sender="0x" + "22" * 20).deliver)
    box.put(event(0))
    assert box.deliver_due() == 0
    assert box.counts() == {"pending": 1, "dead": 0} and box.backoff == 1.0


def test_contract_address_from_deployed_file(tmp_path):
    f = tmp_path / "deployed.json"
    f.write_text(json.dumps({"address": "0xeae6327b50c44b3ea957c97dca44283e75bb1f84", "abi": []}))
    assert load_contract_address(deployed_file=str(f)) == "0xeAe6327b50c44b3ea957C97DCA44283E75BB1F84"
    assert load_contract_address("0xeae6327b50c44b3ea957c97dca44283e75bb1f84", str(f)).startswith("0xeAe6")
