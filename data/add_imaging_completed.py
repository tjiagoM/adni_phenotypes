"""
Script to add a variable to indicate whether a subject is in the imaging cohort
"""

import sqlite3
import csv

# connect to SQL database
CONNECTION = sqlite3.connect("/home/tim/codeLibrary/adni_phenotypes/data/ukb40183.db")
CUR = CONNECTION.cursor()

# import subject list
RESULTS_FILE = "/home/tim/codeLibrary/adni_phenotypes/results/latest_output_ukb_50.csv"
RESULTS_OPEN = open(RESULTS_FILE)
RESULTS_READER = csv.DictReader(RESULTS_OPEN,
                                delimiter=",")
SUBJ_LIST_IMG = [line["ukb_id"] for line in RESULTS_READER]

# convert eids to local
BRIDGE = "bridge_46620_20904_20200122.csv"
BRIDGE_OPEN = open(BRIDGE)
BRIDGE_READER = csv.DictReader(BRIDGE_OPEN, delimiter=",")
BRIDGE_DICT = {bline["eid20904"]: bline["eid46620"]
               for bline in BRIDGE_READER}

SUBJ_LIST = [BRIDGE_DICT[subj] for subj in SUBJ_LIST_IMG]

# add variable to baseline table
# add column
CUR.execute("ALTER TABLE baseline ADD COLUMN imaging DEFAULT 0")
CONNECTION.commit()

for SUBJ in SUBJ_LIST:
    CUR.execute(' '.join(["UPDATE baseline SET imaging = 1",
                          "WHERE eid = '" + SUBJ + "'"]))
    CONNECTION.commit()

CONNECTION.close()
RESULTS_OPEN.close()
