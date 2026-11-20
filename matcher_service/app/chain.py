"""Writes attendance straight to the `Attendance` smart contract (../blockchain attendance/contracts/attendance.sol).

    markAttendance(string memberID)   onlyOwner; the record's time is block.timestamp

Only the contract owner (the deploying account) may write. Two ways to send:
  * CHAIN_PRIVATE_KEY set: the transaction is signed here and sent raw (works with any node);
  * otherwise: eth_sendTransaction from CHAIN_FROM, or the node's first unlocked account (Ganache).

Idempotency: the tx hash is checkpointed in the outbox right after sending, so after a timeout or a
restart the same transaction is looked up again instead of writing the member twice.
"""
import json
import logging
from pathlib import Path

from web3 import Web3
from web3.exceptions import TransactionNotFound

from .sinks import DeliveryRejected, DeliveryRetry

log = logging.getLogger("matcher.chain")

# The subset of the deployed ABI the matcher uses (same as deployed.json written by deploy.py).
ATTENDANCE_ABI = [
    {"inputs": [{"internalType": "string", "name": "memberID", "type": "string"}],
     "name": "markAttendance", "outputs": [], "stateMutability": "nonpayable", "type": "function"},
    {"inputs": [], "name": "getRecordCount",
     "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}], "stateMutability": "view", "type": "function"},
    {"inputs": [], "name": "owner",
     "outputs": [{"internalType": "address", "name": "", "type": "address"}], "stateMutability": "view", "type": "function"},
    {"anonymous": False, "name": "AttendanceMarked", "type": "event",
     "inputs": [{"indexed": False, "internalType": "string", "name": "memberID", "type": "string"},
                {"indexed": False, "internalType": "uint256", "name": "timestamp", "type": "uint256"}]},
]


def load_contract_address(address=None, deployed_file=None):
    """CHAIN_CONTRACT_ADDRESS wins; otherwise the "address" of a deploy.py deployed.json."""
    if address:
        return Web3.to_checksum_address(address)
    if deployed_file:
        return Web3.to_checksum_address(json.loads(Path(deployed_file).read_text())["address"])
    raise RuntimeError("set CHAIN_CONTRACT_ADDRESS or CHAIN_DEPLOYED_FILE")


def make_web3(rpc_url, timeout=10.0, tunnel_token=None):
    headers = {"Content-Type": "application/json"}
    if tunnel_token:
        # Private Microsoft dev tunnel (devtunnels.ms): `devtunnel token <id> --scopes connect`
        headers["X-Tunnel-Authorization"] = f"tunnel {tunnel_token}"
    return Web3(Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": timeout, "headers": headers}))


class ChainSink:
    name = "chain"

    def __init__(self, w3, contract_address, private_key=None, sender=None, receipt_timeout=60.0):
        self.w3 = w3
        self.contract = w3.eth.contract(address=Web3.to_checksum_address(contract_address), abi=ATTENDANCE_ABI)
        self.receipt_timeout = receipt_timeout
        self._account = w3.eth.account.from_key(private_key) if private_key else None
        self._sender = Web3.to_checksum_address(sender) if sender else None
        self._chain_id = None

    @property
    def sender(self):
        if self._account is not None:
            return self._account.address
        if self._sender is None:
            self._sender = self.w3.eth.accounts[0]      # Ganache: first unlocked account = deployer
        return self._sender

    def check(self):
        """Startup sanity check (raises on connection problems). Returns a summary dict."""
        if not self.w3.eth.get_code(self.contract.address):
            raise RuntimeError(f"no contract deployed at {self.contract.address} (redeploy and update the address?)")
        owner = self.contract.functions.owner().call()
        info = {"chain_id": self.w3.eth.chain_id, "contract": self.contract.address, "sender": self.sender,
                "owner": owner, "records": self.contract.functions.getRecordCount().call()}
        if owner.lower() != self.sender.lower():
            log.error("sender %s is not the contract owner %s: every markAttendance will revert", self.sender, owner)
        return info

    def deliver(self, payload, progress=None, checkpoint=None):
        member_id = payload.get("member_id")
        if not member_id:
            raise DeliveryRejected("event has no member_id (imposters are not written to the chain)")
        try:
            if progress:
                result = self._resume(progress)
                if result is not None:
                    return result
            tx_hash = self._send(str(member_id))
            if checkpoint is not None:
                checkpoint(tx_hash)
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=self.receipt_timeout, poll_latency=0.5)
            return self._result(tx_hash, receipt)
        except (DeliveryRetry, DeliveryRejected):
            raise
        except Exception as exc:     # node down, tunnel 401, timeout, revert (wrong account)...: try again later
            raise DeliveryRetry(f"{type(exc).__name__}: {exc}") from exc

    def _resume(self, tx_hash):
        """Result if tx_hash was mined, None if the node no longer knows it (resend); retry if still pending."""
        try:
            receipt = self.w3.eth.get_transaction_receipt(tx_hash)
        except TransactionNotFound:
            try:
                self.w3.eth.get_transaction(tx_hash)
            except TransactionNotFound:
                log.warning("transaction %s was dropped by the node; sending again", tx_hash)
                return None
            raise DeliveryRetry(f"transaction {tx_hash} still pending")
        if receipt["status"] != 1:
            return None                                   # reverted earlier: nothing was recorded, send again
        return self._result(tx_hash, receipt)

    def _send(self, member_id):
        fn = self.contract.functions.markAttendance(member_id)
        if self._account is None:
            tx_hash = fn.transact({"from": self.sender})
        else:
            if self._chain_id is None:
                self._chain_id = self.w3.eth.chain_id
            tx = fn.build_transaction({
                "from": self._account.address,
                "nonce": self.w3.eth.get_transaction_count(self._account.address, "pending"),
                "chainId": self._chain_id,
            })
            signed = self._account.sign_transaction(tx)
            raw = getattr(signed, "raw_transaction", None) or signed.rawTransaction    # web3 7+ / 6
            tx_hash = self.w3.eth.send_raw_transaction(raw)
        return Web3.to_hex(tx_hash)

    def _result(self, tx_hash, receipt):
        if receipt["status"] != 1:
            # Reverted (e.g. not the owner). The member was not recorded: retry once the config is fixed.
            raise DeliveryRetry(f"transaction {tx_hash} reverted")
        return {"status": "recorded", "tx_hash": tx_hash, "block_number": receipt["blockNumber"]}
