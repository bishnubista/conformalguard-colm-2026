# Third-Party Data and Privacy

The original source code is covered by `LICENSE`. Dataset terms are separate:
the code license does not relicense any email corpus.

| Corpus | Upstream | Terms relevant to this export |
|---|---|---|
| Nazario Phishing Corpus | <https://monkey.org/~jose/phishing/> | The current upstream collection includes a CC BY 4.0 notice. It contains real reported phishing messages, and later snapshots may not be anonymized. |
| Phishing Pot | <https://github.com/rf-peixoto/phishing_pot> | The pinned upstream repository is CC BY-NC 4.0 and contains real honeypot messages. |
| SpamAssassin Public Corpus | <https://spamassassin.apache.org/old/publiccorpus/> | The upstream readme says original senders retain copyright and warns against using the messages in a live email system. |
| Enron-Spam AUEB | <https://www2.aueb.gr/users/ion/data/enron-spam/> | Derived from the public Enron corpus and distributed for research. Messages can contain names, addresses, and corporate communications. |

For privacy and licensing caution, the camera-ready public export does not
redistribute raw messages or parsed body text. It includes scripts, checksums,
source-prefixed content IDs, labels, scores, and aggregate outputs. Researchers
who download the upstream corpora are responsible for reviewing the applicable
terms, securing the data, and avoiding attempts to identify or contact message
subjects.
