# Face Recognition Attendance System with Blockchain

> A secure, automated attendance management system using facial recognition and blockchain technology.

**Mubanga Muma** | BSc Computer Science | 10-Week Build

---

## Overview

This system replaces manual attendance methods with a fully automated pipeline. A camera captures a live feed, the face recognition engine identifies registered individuals, and their attendance is recorded to a tamper-proof blockchain ledger. Administrators can view, audit, and export records through a Django web dashboard.

**Three core components:**

- **Face Recognition Engine** — Python + OpenCV (Haar Cascade detection + LBPH recognition)
- **Blockchain Layer** — Ganache local Ethereum testnet + Solidity smart contract
- **Admin Dashboard** — Django backend + frontend UI

---

## Tech Stack

| Component | Tool / Tech | Notes |
|---|---|---|
| Face Detection | OpenCV + Haar Cascade | Real-time detection from webcam/CCTV |
| Face Recognition | OpenCV LBPH | Identifies registered individuals |
| Blockchain | Ganache + Solidity | Local Ethereum testnet |
| Python-Blockchain Bridge | web3.py | Connects Python to the smart contract |
| Local Database | SQLite | Operational queries and reporting |
| Backend Framework | Django | Admin dashboard + REST API |
| Frontend | Django Templates / React | Admin UI |
| Version Control | Git + GitHub | Branching: main, dev, feature/* |
| CI Pipeline | GitHub Actions | Lint + test on every push |
| Deployment | Docker + Render.com | Containerised, free tier hosting |

---

## Repository Structure

```
face-recognition-attendance/
├── .github/
│   └── workflows/
│       └── ci.yml                  # GitHub Actions CI pipeline
├── recognition/
│   ├── detector.py                 # Haar Cascade face detection
│   ├── recognizer.py               # LBPH face recognition
│   ├── trainer.py                  # Train model on registered faces
│   └── dataset/                    # Training images (gitignored)
├── blockchain/
│   ├── contracts/
│   │   └── Attendance.sol          # Solidity smart contract
│   ├── deploy.py                   # Deploy script (web3.py)
│   └── attendance_chain.py         # Python interface to contract
├── dashboard/
│   ├── manage.py
│   ├── dashboard/                  # Django app
│   │   ├── views.py
│   │   ├── models.py
│   │   └── templates/
│   └── requirements.txt
├── tests/
│   ├── test_recognition.py
│   ├── test_blockchain.py
│   └── test_dashboard.py
├── Dockerfile
├── docker-compose.yml
├── .env.example                    # Environment variable template
└── README.md
```

---

## 10-Week Build Plan

### Week 1–2 — Face Recognition
- Learn OpenCV basics (VideoCapture, Haar Cascade, LBPH)
- Implement face detection from webcam feed
- Collect training images and train the LBPH recognizer
- Test recognition accuracy

**Deliverable:** Working face recognition script

---

### Week 3–4 — Blockchain
- Learn blockchain fundamentals (blocks, hashes, chains, Ethereum accounts)
- Set up Ganache local Ethereum testnet
- Write Solidity smart contract for attendance recording
- Deploy and test contract using Remix IDE

**Deliverable:** Deployed smart contract on local chain

---

### Week 5–6 — Integration
- Connect Python to Ganache via web3.py
- Trigger blockchain write when a face is recognised
- Set up SQLite for local operational data
- Write unit tests for recognition + blockchain modules

**Deliverable:** End-to-end attendance recording working

---

### Week 7–8 — Admin Dashboard
- Set up Django project and app structure
- Build attendance view and member registration UI
- Connect dashboard to SQLite + blockchain read
- Set up GitHub Actions CI pipeline (lint + test)

**Deliverable:** Functional admin dashboard

---

### Week 9–10 — DevOps, Testing & Docs
- Write integration tests and fix bugs
- Dockerize the application
- Deploy to Render.com
- Write final documentation and README
- Prepare presentation demo

**Deliverable:** Deployed system + final docs

---

## DevOps

### Git Branching Strategy

```
main        ← production-ready code only, never pushed to directly
dev         ← integration branch, all features merge here first
feature/*   ← short-lived branches per feature
```

**Workflow:**
1. Create a `feature/` branch off `dev`
2. Build and test locally
3. Push and open a Pull Request into `dev`
4. Review and merge
5. Merge `dev` into `main` when a phase is complete

### CI Pipeline (GitHub Actions)

Every push triggers:
- `flake8` — Python lint check
- `pytest` — unit tests
- Docker build check (optional, added in Week 9)

### Docker

Added in Week 9. A `Dockerfile` and `docker-compose.yml` package the Django app, OpenCV module, and SQLite database together. The same container runs locally and on Render.com.

---

## Smart Contract

```solidity
// Attendance.sol
contract Attendance {
    struct AttendanceRecord {
        address registeredBy;
        string memberID;
        uint256 timestamp;
    }

    AttendanceRecord[] public records;

    function markAttendance(string memory memberID) public {
        records.push(AttendanceRecord(msg.sender, memberID, block.timestamp));
        emit AttendanceMarked(memberID, block.timestamp);
    }

    event AttendanceMarked(string memberID, uint256 timestamp);
}
```

Intentionally minimal. The complexity is in connecting it to the Python recognition engine, not in the contract itself.

---

## Learning Resources

| Topic | Resource |
|---|---|
| OpenCV | [Official OpenCV Python Tutorials](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html) |
| Solidity / Blockchain | [CryptoZombies](https://cryptozombies.io) — Lessons 1 and 2 are enough |
| web3.py | [web3.py Quickstart Docs](https://web3py.readthedocs.io/en/stable/quickstart.html) |
| Ganache | [Ganache Documentation](https://trufflesuite.com/docs/ganache/) |

---

## Definition of Done

The project is complete when:

- [ ] A registered member's face is detected and attendance is recorded automatically
- [ ] The attendance record is written to the Ganache blockchain and can be read back
- [ ] An unregistered face triggers a flagged / unknown entry
- [ ] The Django admin dashboard displays records from both SQLite and the blockchain
- [ ] All tests pass in the GitHub Actions CI pipeline
- [ ] The application runs in a Docker container and is deployed to Render.com
- [ ] A README and project report document the architecture, setup steps, and known limitations

---

## Known Limitations

- Recognition accuracy degrades in poor lighting conditions
- Ganache is a local testnet — not a live public blockchain
- Facial data storage raises privacy considerations (encodings only, no raw images stored)
- Real-time processing requires reasonable hardware specs

---

## License

Academic project — BSc Computer Science, 2026.
