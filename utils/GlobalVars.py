"""
/*
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 */
"""
LOG_DIR = "./log"
RESULTS_DIR = "./results"
SAVED_DIR = "./saved"
DATASETS_DIR = ".datasets/"

GENERAL_MODELS = ["Pop", "ItemKNN", "BPR", "NeuMF", "ConvNCF", "DMF", "FISM", "NAIS", "SpectralCF", "GCMC",
                  "NGCF", "LightGCN", "DGCF", "LINE", "MultiVAE", "MultiDAE", "MacridVAE", "CDAE", "ENMF",
                  "NNCF", "RaCT", "RecVAE", "EASE", "SLIMElastic"]

CONTEXT_MODELS = ["LR", "FM", "NFM", "DeepFM", "xDeepFM", "AFM", "FFM", "FwFM", "FNN", "PNN", "DSSM", "WideDeep",
                  "DCN", "AutoInt"]

KNOWLEDGE_MODELS = ["CKE", "CFKG", "KTUP", "KGAT", "RippleNet", "MKR", "KGCN", "KGNNLS"]

SEQUENTIAL_MODELS = ["FPMC", "GRU4REC", "NARM", "STAMP", "Caser", "NextItNet", "TransRec", "SASRec", "BERT4Rec",
                     "SRGNN", "GCSAN", "GRU4RecF", "SASRecF", "FDSA", "S3Rec", "GRU4RecKG", "KSR", "FOSSIL",
                     "SHAN", "RepeatNet", "HGN", "HRM", "NPE"]
