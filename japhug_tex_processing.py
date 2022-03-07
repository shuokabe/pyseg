import os
import re


def check_dagger(string):
    '''Look for the $dagger$ symbol in the string.'''
    if (string.find('$\dagger$') >= 0) or (string.find('†') >= 0):
        return True
    else:
        return False

def check_other_language(string):
    '''Look for characters from other languages.'''
    if (string.find('ʔ') >= 0) or (string.find('ɐ') >= 0) \
        or (string.find('ə') >= 0):
        return True
    else:
        return False

def check_citation(string):
    '''Look for the \cite command in the string.'''
    if string.find(r'\cite') >= 0:
        return True
    else:
        return False

def extract_element(file, file_name, start, end):
    '''Extract all the desired data between indices'''
    element_list = []
    i = start
    while (i < end):
        line = file[i].lstrip() # To avoid issues with leading whitespace
        #print(f'This is line {i}: {line}')
        if line.startswith('\gll'):
            gll = line
            gloss = f'\glo {file[(i + 1)].lstrip()}'
            j = i + 2
            glt = file[j].lstrip() #lioness issue
            com = f'\com File {file_name}, line {i}'
            complex = False # Complex cases (nested, shared translation, etc.)
            if check_dagger(gll) or check_other_language(gll):
                pass
            else:
                # Find the correct line
                while (not glt.startswith('\glt')) and (j <= end):
                    #print('GLT', glt)
                    if glt.startswith('\gll'):
                        print(f'Error: glt not found (?), line {j}')
                        complex = True
                    j += 1
                    glt = file[j].lstrip()
                if glt.startswith('\glt'):
                    if check_citation(glt):
                        pass
                    else:
                        #element_list.append([gll, gloss, glt, com])
                        four_elements = '\n'.join([gll, gloss, glt, com])
                        element_list.append(four_elements)
                else:
                    print(f'Line {j} is not a glt line.')
        #glt = file.index('\glt')
            if complex:
                i += 2
            else:
                i = j
        else:
            i += 1
    return element_list

def remove_head_tail_space(split_text):
    '''Remove leading and trailing whitespaces.'''
    return [sentence.strip() for sentence in split_text]

def find_index_list(target_list, element, start_index):
    '''Find the index of an element in a list, starting from a given index.'''
    if element in target_list[start_index:]:
        return target_list.index(element, start_index)
    else:
        return -1

def data_from_file(file, file_name):
    '''Extract data element from the tex file.'''
    split_file = re.split('\n', file)
    split_file = remove_head_tail_space(split_file)
    data_list = []
    n = len(split_file)
    i = 0
    while (i < n) and (i >= 0):
        start_exe = find_index_list(split_file, r'\begin{exe}', i)
        end_exe = find_index_list(split_file, r'\end{exe}', start_exe) #i)
        assert start_exe <= end_exe, (f'At {i}, {start_exe} should be '
                                      f'below {end_exe}')
        #print(f'Start {start_exe}, end {end_exe}')
        if (start_exe == -1) and (end_exe == -1):
            i = -1
        elif (start_exe == -1) or (end_exe == -1):
            print(f'Error: start_exe = {start_exe} and end_exe = {end_exe}')
        else:
            data_list.extend(extract_element(split_file, file_name,
                                             start_exe, end_exe))
            i = end_exe + 1
    return data_list
    #for element in text_root.iter('S'):
    #    sentence = element.find('FORM').text
    #    sentence_list.append(sentence)
    #if sentence_list == []: # For already processed files
    #    text = text_root.find('FORM').text
    #    split_text = re.split('\n', text)
    #    sentence_list = [line for line in split_text if line != '']
    #return sentence_list

# tex files in the folder
def list_tex_files(path):
    '''List the tex files in the folder.'''
    file_list = []
    for file in os.listdir(path):
        if file.endswith('.tex'):
            print(file)
            file_list.append(file)
    print(f'{len(file_list)} files in the folder\n')
    file_list.sort() # Sort
    return file_list

# Creation of the structured corpus
def create_corpus(path):
    '''Create structured corpus'''
    file_list = list_tex_files(path)
    data_list = []
    for file in file_list:
        print(f'File: {file}')
        raw_text = open(os.path.join(path, file), 'r').read()
        data_list.extend(data_from_file(raw_text, file))
        print(f'Number of elements so far: {len(data_list)}')
    return '\n\n'.join(data_list)

def export_text(corpus_string, name='corpus'):
    '''Save a string containing a text in a txt file.'''
    output_file = name + '.txt'
    with open(output_file, 'w', encoding = 'utf8') as out_text:
        out_text.write(corpus_string)

# Create a file with a specific field only
def extract_one_field(corpus, field, name):
    raw_corpus = open(corpus, 'r').read()
    split_corpus = re.split('\n', raw_corpus)
    desired_field = [line for line in split_corpus if line.startswith(field)]
    export_text('\n'.join(desired_field), name)

# Preprocessing for \gll
def remove_inbetween_dollar(string):
    '''Remove whatever is between two dollar signs.'''
    return re.sub('[$][^$]+[$]', '', string)

def remove_backslash_latex(string):
    '''Remove latex commands of the form: \command{...}.'''
    return re.sub('[{]([^}]+)[}]', r'\1', string)

def remove_excessive_whitespace(string):
    '''Remove excessive whitespace.'''
    return re.sub(' +', ' ', string)

def strip_whole_text(text):
    '''Strip whitespace from each sentence.'''
    split_text = re.split('\n', text)
    return '\n'.join([sentence.strip() for sentence in split_text])

def pre_process_gll(text, morpheme=0):
    '''Preprocess text for \gll content'''
    # Doubts:
    # -\glll entries
    # - <>, (), [], latex commands, special characters (^)
    # % symbol, [...], = symbol
    text_glll = text.replace(r'\glll ', r'\gll ')
    text_gll = text_glll.replace(r'\gll ', '')
    text_2backslash = text_gll.replace(r'\\', '')
    removed_text = remove_inbetween_dollar(text_2backslash)
    commands_to_remove = [r'\bleu', r'\rouge', r'\textbf', r'\phantom{espace}']
    for command in commands_to_remove:
        removed_text = removed_text.replace(command, '')
    removed_text = removed_text.replace(r'\redp{}', '-') # Reduplication
    removed_text = re.sub('[TAL]', '', removed_text) # Conversation speaker
    pp_text = remove_excessive_whitespace(removed_text)
    pp_text = re.sub(r'[!?:…,.()<>\t"]', '', pp_text)
    pp_text = re.sub("['́'`'̀''̂%]", '', pp_text)
    pp_text = pp_text.replace('ɡ', 'g')
    pp_text = pp_text.replace('\gll\n', '')
    pp_text = re.sub('[\{\}\[\]]', '', pp_text)
    pp_text = pp_text.replace('/', ' ')
    pp_text = pp_text.replace('\\t', ' ')
    pp_text = pp_text.lower() # Dai Song case
    pp_text = remove_excessive_whitespace(pp_text)
    if morpheme == 2:
        final_text = re.sub("[-=]", '-', pp_text)
    elif morpheme == 1:
        final_text = re.sub("[-=]", ' ', pp_text)
    else:
        final_text = re.sub("[-=]", '', pp_text)
    return remove_excessive_whitespace(strip_whole_text(final_text))
