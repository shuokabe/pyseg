# pyseg

pyseg is a Python implementation of dpseg, a word segmentation model, and its variant, pypseg.

dpseg is a Dirichlet process word segmentation model, developed by Sharon Goldwater and explained in [A Bayesian Framework for Word Segmentation: Exploring the Effects of Context][1], Sharon Goldwater, Thomas L. Griffiths, and Mark Johnson.

The original implementation in C++ can be found [here][2].

[1]: https://homepages.inf.ed.ac.uk/sgwater/papers/cognition-hdp.pdf
[2]: https://homepages.inf.ed.ac.uk/sgwater/resources.html


## System requirements
pyseg runs on Python 3 (tested on Python 3.6 and 3.8).

## How to use?

### Parameters
Here are the parameters that can be used:
- `filename`: the path to file (must be specified)
- `model` (or -m): the model name (dpseg or pypseg; dpseg by default)
- `alpha_1` (or -a): the concentration parameter value for unigram DP (20 by default)
- `p_boundary` (or -b): the prior probability of word boundary (0.5 by default)
- `discount` (or -d): the discount parameter for the PYP model (0.5 by default)
- `iterations` (or -i): the number of iterations (100 by default)
- `output_file_base` (or -o): the output base filename (output by default, i.e. the output file will be called output.txt)
- `rnd_seed` (or -r): the random seed (42 by default)
- `sample_hyperparameter` (or -s): the boolean parameter for hyperparameter sampling (False by default)

Supervision parameter arguments:
- `supervision_file`: the file name of the data used for supervision (pickle dictionary)
- `supervision_method`: the supervision method (word dictionary) (choose between naive, naive_freq, mixture, mixture_bigram, initialise, init_bigram; none by default)
- `supervision_parameter`: the parameter value for (dictionary) supervision (0 by default)
- `supervision_boundary`: the boundary supervision method (choose between true, random, sentence, word; none by default)
- `supervision_boundary_parameter`: the parameter value for boundary supervision (0 by default)
- `online`: online learning method (without, with, bigram; none by default)
- `online_batch`: the number of sentences after which Gibbs sampling is carried out for online learning (0 by default)
- `online_iter`: the number of iterations for online learning (0 by default)

Two-level segmentation arguments:
- `alpha_m` (or -am): the concentration parameter value for the morpheme level in hierarchical models (20 by default)
- `discount_m` (or -dm): the discount parameter for the morpheme level in hierarchical models (0.5 by default)
- `htl_level`: the supervision level for the hierarchical models (none by default)
- `htl_batch`: the batch size for the hier_iter model (0 by default)
- `htl_iter`: the number of additional iterations for the morpheme level in the hier_iter model (0 by default)
- `just_seg`: the boolean parameter to carry out segmentation only (without evaluation) (False by default)

Other:
- `verbose` (or -v): verbosity of the output (False by default)
- `version`: version number

### Example

To run dpseg on a file called file.txt for 1,000 iterations with a seed of 42, the terminal command is as follows:

```
python3 pyseg/main.py -m dpseg -i 1000 -r 42 file.txt
```

If a dpseg model is run on the same file (file.txt) and with the same parameters but with a mixture supervision of parameter 0.25 and the dictionary file dictionary.pickle, the command changes to:
```
python3 pyseg/main.py -m dpseg -i 1000 -o sup_pydpseg -r 42 --dictionary_file dictionary.pickle --dictionary_method mixture --dictionary_parameter 0.25 file.txt
```
Moreover, here, the output file will not be called output.txt but sup_pydpseg.txt.

## Name correspondence
The names in the paper and in the code do not match (yet). The list below displays the corresponding name in the code for a given name from the paper (name_in_paper: equivalent_in_code).

Models:
- parallel-w/parallel-m: two_level
- hier_type: sim_htl
- hier_iter: sim_htl with positive values of parameters htl_batch and htl_iter
- pipeline and htl_final: use of the `just_seg` parameter

Weak supervision:
- gs.sparse: supervision_boundary set to true
- gs.dense: supervision_boundary set to sentence
- d.count: supervision_method set to naive
- d.mix: supervision_method set to mixture
- d.2gram: supervision_method set to init_bigram
- d.mix+2gram: supervision_method set to mixture_bigram

Online learning:
- o.regular: with
- o.2level: bigram

## Citation
If you use pyseg, please use the following citation (from ACL Anthology):
```
@inproceedings{Okabe22weakly,
	title = "Weakly Supervised Word Segmentation for Computational Language Documentation",
    author = "Okabe, Shu  and
      Besacier, Laurent  and
      Yvon, Fran{\c{c}}ois",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-long.510",
    pages = "7385--7398"
}
```
