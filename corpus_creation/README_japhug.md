# Japhug corpus

The tex files of [(Jacques, 2021)][1] can be found on [this Github link][2].

[1]: https://langsci-press.org/catalog/book/295
[2]: https://github.com/langsci/295/tree/main/chapters

## Creating the corpus
Once all files are downloaded, the japhug_tex_processing.py file contains the pre-processing function.

### Raw extract from the tex files
In Python, the following code extracts all tiers in the examples:
```
import japhug_tex_processing as tp

japhug_corpus = tp.create_corpus('295/chapters/')
```
where '295/chapters' is the path of the tex files (here, the original architecture).

The output will have units composed of four tiers, as such:
- `\gll`: the segmented transcription (' ' for word boundaries and '-' for morpheme boundaries)
- `\glo`: the glosses for the sentence
- `\glt`: the English translation of the sentence
- `\com`: the source of the sentence, in the following format: 'File file_name, line line_number'

It is **important** to note that here, the units are **raw** extracts from the tex files and need to be pre-processed.
This file can also be used to create corpora for other tasks, such as glossing or translation.

```
tp.export_text(japhug_corpus, 'tex_japhug_corpus')
```
This saves the raw corpus in a file named 'tex_japhug_corpus.txt'.


### Extracting the segmented sentences (`\gll`) only
The following command creates a file called 'raw_japhug_gll_corpus.txt' made of `\gll` tiers for all sentences from the 'tex_japhug_corpus.txt' file.
```
tp.extract_one_field('tex_japhug_corpus.txt', '\gll', 'raw_japhug_gll_corpus')
```


### Pre-processing the sentences
Finally, the following code creates the pre-processed file with sentences segmented at the word level.
```
raw_text = open('raw_japhug_gll_corpus.txt', 'r').read()
pp_text = tp.pre_process_gll(raw_text, morpheme = 0)
```

The `pre_process_gll` function contains a `morpheme` parameter that makes creating texts with different segmentation levels possible.
- with `morpheme = 0`, the corpus is segmented at the word level only (' ' between words)
- with `morpheme = 1`, the corpus is segmented at the morpheme level only (' ' between all morphemes, even between words)
- with `morpheme = 2`, the corpus is segmented at the two levels, ' ' between words and '-' between morphemes.

The following table illustrates the three segmentation possibilities for a sentence (*There were three brothers* in English):
| morpheme =  | segmentation level | Japhug sentence |
| --- | --- | --- |
| 0 | word | kɤndʑixtɤɣ χsɯm pjɤtunɯ |
| 1 | morpheme | kɤndʑi xtɤɣ χsɯm pjɤ tu nɯ |
| 2 | two-level | kɤndʑi-xtɤɣ χsɯm pjɤ-tu-nɯ |

Finally, specifying `encoding = utf8` when saving is recommended because of the special characters.
```
with open('pp_tex_japhug_corpus.txt', 'w', encoding = 'utf8') as out_text:
    out_text.write(pp_text)
```
