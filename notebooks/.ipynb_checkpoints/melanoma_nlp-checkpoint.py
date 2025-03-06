import pandas as pd
import numpy as np
import medspacy
import spacy
from medspacy.ner import TargetRule
from spacy.tokens import Span, Token
from medspacy.context import ConText, ConTextRule
from medspacy.preprocess import Preprocessor, PreprocessingRule
import re
import os
from medspacy.ner import ConceptTagger
import json
from spacy.util import compile_infix_regex, compile_prefix_regex,compile_suffix_regex
from spacy.tokenizer import Tokenizer
from spacy.matcher import Matcher
from spacy.language import Language

try:
    import pyodbc
except:
    pass


#Used to specify default entity types for context
ALLOWED_TYPES = ["HISTOLOGY","MEL_UNSPEC","MALIGNANCY","HISTOLOGY_TYPE","TUMOR_HIST","PENDING","NOT_MELANOMA_DX","TUMOR_UNSPEC"]

#Default context scope (otherwise specified for the rule)
CONTEXT_SCOPE = 10

#Used to replace carriage returns and |tab| strings before processessing (These can confuse displayed text and/or rules)
preprocess_rules = [ #replace carriage return with newline for ease of viewing and consistency
    PreprocessingRule(
        "\r",
        repl ="\n",
    ),
    PreprocessingRule(
        "\|TAB\|",
        repl ="     ",
    )
]

#This is a pattern used to tag cases where multiple concepts are given a result in path notes
LIST_TAGGER_PATT = [
        [
            #{"LOWER":{"REGEX":"[a-z]*"},"OP":"{2,}"},
            {
                "_":{"concept_tag":{"NOT_IN":["ABSENT_BEFORE","ABSENT","INDETERMINATE","PRESENT","IMPENDING","ULCERATION"]}},
                "IS_PUNCT":False,
                "OP":"{1,4}"
            },
            {'IS_SPACE': True, 'OP': '*'},
            {"TEXT":","},
            {
                "_":{"concept_tag":{"NOT_IN":["ABSENT_BEFORE","ABSENT","INDETERMINATE","PRESENT","IMPENDING"]}},
                "IS_PUNCT":False,
                "OP":"{1,4}"
            },
            {'IS_SPACE': True, 'OP': '*'},
            {"TEXT":","}
        ],
        [
            #{"LOWER":{"REGEX":"[a-z]*"},"OP":"{2,}"},
            {
                "_":{"concept_tag":{"NOT_IN":["ABSENT_BEFORE","ABSENT","INDETERMINATE","PRESENT","IMPENDING","ULCERATION"]}},
                "IS_PUNCT":False,
                "OP":"{1,4}"
            },
            {'IS_SPACE': True, 'OP': '*'},
            {"TEXT":","},
            {
                "_":{"concept_tag":{"NOT_IN":["ABSENT_BEFORE","ABSENT","INDETERMINATE","PRESENT","IMPENDING"]}},
                "IS_PUNCT":False,
                "OP":"{1,4}"
            },
            {'IS_SPACE': True, 'OP': '*'},
            {"LOWER":{"IN":["and","or","nor"]}}
        ],
        [
            {"TEXT":","},
            {'IS_SPACE': True, 'OP': '*'},
            {
                "_":{"concept_tag":{"NOT_IN":["ABSENT_BEFORE","ABSENT","INDETERMINATE","PRESENT","IMPENDING"]}},
                "IS_PUNCT":False,
                "OP":"{1,4}"
            },
            {'IS_SPACE': True, 'OP': '*'},
            {"TEXT":","}
        ],
        [
            #{"LOWER":{"REGEX":"[a-z]*"},"OP":"{2,}"},
            {
                "_":{"concept_tag":{"NOT_IN":["ABSENT_BEFORE","ABSENT","INDETERMINATE","PRESENT","IMPENDING"]}},
                "IS_PUNCT":False,
                "OP":"{1,4}"
            },
            {'IS_SPACE': True, 'OP': '*'},
            {"TEXT":",","OP":"?"}, #oxford comma
            {'IS_SPACE': True, 'OP': '*'},
            {"LOWER":{"IN":["and","or","nor"]}},
            {'IS_SPACE': True, 'OP': '*'},
            {
                "_":{"concept_tag":{"NOT_IN":["ABSENT_BEFORE","ABSENT","INDETERMINATE","PRESENT","IMPENDING","ULCERATION"]}},
                "IS_PUNCT":False,
                "OP":"{1,4}"
            }
        ]
    ]

def table_import(clean_query: str,db_server: str, db_db: str) -> pd.DataFrame:
    """
    Imports table from SQL.
    
    Input: SQL-formatted string query for the target table.
    
    Returns: Dataframe of table
    """
    db_conn_str = 'DRIVER={ODBC Driver 17 for SQL Server}'\
                          +';SERVER='+db_server\
                          +';DATABASE='+db_db\
                          +';TRUSTED_CONNECTION=yes'
    try:
        sql_con = pyodbc.connect(db_conn_str)
        df = pd.read_sql(clean_query, sql_con)
        sql_con.close()
    except Exception as e:
        print(f"An error occurred: {e}")
        try: 
            sql_con.close()
        except:
            pass
        return
    return df


def csv_to_targetmatcher(filepath=None,insert_whitespace_matching=False):
    """Function to pull rules from a csv file. insert_whitespace_matching=True will cause a regex match for whitespace to be inserted between each token match rule"""
    import pandas as pd
    from medspacy.ner import TargetRule
    import ast
    df = pd.read_csv(filepath,encoding='ISO-8859-1',escapechar=None)
    target_rules = []
    for i,row in df.iterrows():
        patt = None
        try: 
            patt = ast.literal_eval(row['pattern'])#ast.literal_eval(repr(row['pattern'])[1:-1])##ast.literal_eval(repr(row['pattern'])[1:-1])#fix backslashes
            if insert_whitespace_matching:
                #patt = [x for item in patt for x in (item,{"IS_SPACE":True,"OP":"*"})][:-1]
                #whitespace_dict = {"REGEX":"^[^\na-zA-Z]$"}
                #patt = [x for item in patt for x in (item,{"TEXT":{"REGEX":re.compile(f"{test_dict['REGEX']}").pattern}})][:-1]
                patt = [x for item in patt for x in (item,{"TEXT":{"REGEX":r'^[^\S\n]+$'},"OP":"*"},{"TEXT":{"REGEX":r'^[^\S\n]*\n[^\S\n]*$'},"OP":"*"},{"TEXT":{"REGEX":r'^[^\S\n]+$'},"OP":"*"})][:-3]
        except:
            #print(row['pattern'], ' appears to be a list, but literal_eval was unable to parse as a list of dictionaries. It is being used a string')
            if (str(row['pattern']) != 'nan') and (str(row['pattern']) != 'None'):
                patt = [{'LOWER':{'REGEX':row['pattern']}}]#str(repr(str(row['pattern']))[1:-1])#fix backslashes
        if (str(row['label']) != 'nan') and (str(row['label']) != 'None'):
            target_rules.append(TargetRule(row['literal'],row['label'],pattern=patt))
    return target_rules


def csv_to_contextrules(filepath=None,insert_whitespace_matching=False):
    """Function to pull rules from a csv file. insert_whitespace_matching=True will cause a regex match for whitespace to be inserted between each token match rule"""
    import pandas as pd
    import ast
    df = pd.read_csv(filepath,encoding='ISO-8859-1',escapechar=None)
    target_rules = []
    for i,row in df.iterrows():
        patt = None
        try: 
            patt = ast.literal_eval(row['pattern'])#ast.literal_eval(repr(row['pattern'])[1:-1])##ast.literal_eval(repr(row['pattern'])[1:-1])#fix backslashes
            if insert_whitespace_matching:
                #patt = [x for item in patt for x in (item,{"IS_SPACE":True,"OP":"*"})][:-1]
                #whitespace_dict = {"REGEX":"^[^\na-zA-Z]$"}
                #patt = [x for item in patt for x in (item,{"TEXT":{"REGEX":re.compile(f"{test_dict['REGEX']}").pattern}})][:-1]
                patt = [x for item in patt for x in (item,{"TEXT":{"REGEX":r'^[^\S\n]+$'},"OP":"*"},{"TEXT":{"REGEX":r'^[^\S\n]*\n[^\S\n]*$'},"OP":"*"},{"TEXT":{"REGEX":r'^[^\S\n]+$'},"OP":"*"})][:-3]
        except:
            #print(row['pattern'], ' appears to be a list, but literal_eval was unable to parse as a list of dictionaries. It is being used a string')
            if (str(row['pattern']) != 'nan') and (str(row['pattern']) != 'None'):
                patt = [{'LOWER':{'REGEX':row['pattern']}}]#str(repr(str(row['pattern']))[1:-1])#fix backslashes
        if (str(row['label']) != 'nan') and (str(row['label']) != 'None'):
            target_rules.append(ConTextRule(literal=row['literal'],category=row['label'],pattern=patt,direction='BIDIRECTIONAL',max_scope=80))
    return target_rules

def targetrules_to_df(targetrules,remove_whitespace_matching=True,include_direction=False,include_scope=False):
    """Function to convert targetrules to dataframe format for storage/viewing"""
    rule_dict = {"literal":[],"label":[],"pattern":[]}
    if include_direction:
        rule_dict['direction'] = []
    if include_scope:
        rule_dict['scope'] = []
    for rule in targetrules:
        rule_dict['literal'].append(rule.literal)
        rule_dict['label'].append(rule.category)
        if isinstance(rule.pattern,list):
            if remove_whitespace_matching:
                rule_dict['pattern'].append(rule.pattern[0::4])
            else:
                rule_dict['pattern'].append(rule.pattern)
        else:
            rule_dict['pattern'].append(None)
        if include_direction:
            rule_dict['direction'].append(rule.direction)
        if include_scope:
            rule_dict['scope'].append(rule.max_scope)
    return pd.DataFrame.from_dict(rule_dict)


def first_token_callback(target,modifier,span_between):
    '''
    Function to remove specific topography matches that don't occur at the beginning of a document. There are alternative matcher rules that work better for later in the document.
    '''
    if modifier.start != 0:
        return False
    else:
        return True

def add_onmatch_terminators(rule_list):
    """Used to make quick adjustments to rules without changing json, or to add arguments to rules that are not saved in json files"""
    context_rules_upd = []
    for rule in rule_list:
        if rule.category == "HISTOLOGY_TYPE":
            context_rules_upd.append(ConTextRule(literal=rule.literal, category=rule.category, pattern=rule.pattern, direction=rule.direction,max_scope=rule.max_scope,max_targets=rule.max_targets,terminated_by=['NONSKIN_TOPOGRAPHY','SKIN_TOPOGRAPHY'],allowed_types=["HISTOLOGY","MEL_UNSPEC"]))
        elif rule.category == "METASTATIC":
            #print("not adjusting met scope")
            context_rules_upd.append(ConTextRule(literal=rule.literal, category=rule.category, pattern=rule.pattern, direction=rule.direction,max_scope=rule.max_scope,max_targets=rule.max_targets,terminated_by=['NONSKIN_TOPOGRAPHY','SKIN_TOPOGRAPHY']))#,allowed_types=["HISTOLOGY","MEL_UNSPEC"]
        elif 'ULCERATION' in rule.category:
            #print("Adding allowed types to ulceration")
            context_rules_upd.append(ConTextRule(literal=rule.literal, category=rule.category, pattern=rule.pattern, direction=rule.direction,max_scope=rule.max_scope,max_targets=rule.max_targets,terminated_by=['NONSKIN_TOPOGRAPHY','SKIN_TOPOGRAPHY']))#,allowed_types=["HISTOLOGY","MEL_UNSPEC"]))
        elif 'CLARK_DEPTH' in rule.category:
            #print("Adding allowed types to clark") ----commented for processing
            context_rules_upd.append(ConTextRule(literal=rule.literal, category=rule.category, pattern=rule.pattern, direction=rule.direction,max_scope=rule.max_scope,max_targets=rule.max_targets,terminated_by=['NONSKIN_TOPOGRAPHY','SKIN_TOPOGRAPHY'],allowed_types=["HISTOLOGY","MEL_UNSPEC"]))#,allowed_types=["HISTOLOGY","MEL_UNSPEC"]
        elif ('DEEP_TRANSECTION' in rule.category) or ('NEGATED_TRANSECTION' in rule.category) or ('UNSPEC_TRANSECTION' in rule.category):
            #print("Adding allowed types to transection")----commented for processing
            context_rules_upd.append(ConTextRule(literal=rule.literal, category=rule.category, pattern=rule.pattern, direction=rule.direction,max_scope=rule.max_scope,max_targets=rule.max_targets,terminated_by=['NONSKIN_TOPOGRAPHY','SKIN_TOPOGRAPHY'],allowed_types=["HISTOLOGY","MEL_UNSPEC",'NOT_MELANOMA_DX']))
        elif (rule.literal in ['Topography_group5_firsttok','Topography_group8_firsttok']):
            context_rules_upd.append(ConTextRule(literal=rule.literal, category=rule.category, pattern=rule.pattern, direction=rule.direction,max_scope=rule.max_scope,max_targets=rule.max_targets,terminated_by=['NONSKIN_TOPOGRAPHY','SKIN_TOPOGRAPHY'],on_modifies=first_token_callback))
        else:
            if rule.direction != 'TERMINATE':
                context_rules_upd.append(ConTextRule(literal=rule.literal, category=rule.category, pattern=rule.pattern, direction=rule.direction,max_scope=rule.max_scope,max_targets=rule.max_targets,terminated_by=['NONSKIN_TOPOGRAPHY','SKIN_TOPOGRAPHY']))
            else:
                print('Excluding Terminate rule: ',rule.literal)
    return context_rules_upd

def build_nlp(file_path='./',csv_or_json_rule_import='json',insert_whitespace_matching=False):
    """
    Builds nlp pipeline using best-presumed nlp pipeline.
    Derives rules for concept_tagger and target_matcher and Context from
        json files found in resource folder.
    Returns full nlp pipeline.
    """
    nlp = medspacy.load(medspacy_disable=["medspacy_pyrush","medspacy_target_matcher","medspacy_context"])#medspacy.load(medspacy_enable=["medspacy_pyrush"])
    
    ###Import rules from json/csv's
    try:
        pretagger_rules = TargetRule.from_json(os.path.join(file_path,'pretagger_rules.json'))
        tagger_rules = TargetRule.from_json(os.path.join(file_path,'tagger_rules.json'))
        target_rules = TargetRule.from_json(os.path.join(file_path,'target_rules.json'))
        context_rules = []
        context_json_list = ['depth_context_rules.json','transect_context_rules.json','mitotic_index_context_rules.json','modifier_context_rules.json','metastasis_context_rules.json','top_hist_context_rules.json','ulceration_context_rules.json','template_exclusion_context_rules.json']
        for json_file in context_json_list:
            context_rules += ConTextRule.from_json(os.path.join(file_path,json_file))
        context_rules = add_onmatch_terminators(context_rules)
    except Exception as e:
        print(f"Except occurred with json import: {e}")
        return

    ###Preprocessor
    
    prefixes = nlp.Defaults.prefixes + [r'\-'] + [r'\n'] #make sure newlines and hyphens are separated during tokenization
    prefix_re = compile_prefix_regex(prefixes)
    nlp.tokenizer.prefix_search = prefix_re.search

    suffixes = nlp.Defaults.suffixes + [r'\-'] + [r'\n']
    suffix_re = compile_suffix_regex(suffixes)
    nlp.tokenizer.suffix_search = suffix_re.search

    preprocessor = Preprocessor(nlp.tokenizer)
    nlp.tokenizer = preprocessor
    
    preprocessor.add(preprocess_rules)

    ###Pretagger

    @Language.component("pretagger")
    def pretagger(doc):
        matcher = Matcher(nlp.vocab)
        pattern = None
        for rule in pretagger_rules:
            if isinstance(rule.pattern,list):
                pattern = rule.pattern
            else:
                pattern = [{"LOWER":rule.literal.lower()}]
            matcher.add(rule.category,[pattern])
        matches = matcher(doc)
        for match_id, start, end in matches:
            span = doc[start:end]
            label = nlp.vocab.strings[match_id]
            for token in span:
                token._.pretag = label
        return doc

    Token.set_extension("pretag",default = "",force=True)
    nlp.add_pipe("pretagger",name="pretagger",last=True)
    
    ###Concept Matcher
    
    concept_matcher = nlp.add_pipe("medspacy_concept_tagger")
    concept_matcher.add(tagger_rules)

    ##List Tagger
    
    @Language.component("list_tagger")
    def list_tagger(doc):
        matcher = Matcher(nlp.vocab)
        pattern = LIST_TAGGER_PATT
        matcher.add("comma_separated_list",pattern)
        matches = matcher(doc)
        for match_id,start,end in matches:
            span = doc[start:end]
            for token in span:
                token._.list_component = True
        return doc
    
    Token.set_extension("list_component",default = False,force=True)
    nlp.add_pipe("list_tagger",name="list_tagger",last=True)
    
    ##Target Matcher
    
    target_matcher = nlp.add_pipe("medspacy_target_matcher")
    target_matcher.add(target_rules)

    ##Context for top/morph/negation

    ##Setting prune on target overlap may want to be True (was like this in validation)
    context = nlp.add_pipe("medspacy_context",config = {"rules":None,"max_scope": CONTEXT_SCOPE, "allowed_types":ALLOWED_TYPES,"prune_on_target_overlap":False})
    context.add(context_rules)

    ##Context for top/morph/negation improved
    #context = ConText(nlp,rules=None,name='context',prune_on_target_overlap=False,max_scope= CONTEXT_SCOPE) #span_attrs=custom_attrs
    #context.add(context_rules)
    #print(context.config)


    #@Language.component("medspacy_context_custom")
    #def thick_ulcer_context(doc):
    #    return context(doc)
    #nlp.add_pipe("medspacy_context_custom",name="medspacy_context_custom",last=True)
    

    ##Context for ulcer/thick/met (sequentially before top context because it uses top as a terminator)
    
    #context_ulcer_thick = ConText(nlp,rules=None,name='ulcer_thick_context',allowed_types=["HISTOLOGY","HISTOLOGY_TYPE","TUMOR_HIST","MEL_UNSPEC","MALIGNANCY"],max_scope=100) #span_attrs=custom_attrs
    #context_ulcer_thick.add(context_ulcer_thick_rules)
    
    #@Language.component("thick_ulcer_context")
    #def thick_ulcer_context(doc):
        #custom_attrs = {'DEPTH_VALUE': {'is_measurement':True}}
    #    modifiers = doc._.context_graph.modifiers
    #    process_doc = context_ulcer_thick(doc)
    #    process_doc._.context_graph.modifiers.extend(modifiers)
    #    return process_doc
    #nlp.add_pipe("thick_ulcer_context",name="thick_ulcer_context",last=True)

    #Cases that indicate a histology type without having melanoma keywords are only meant to be captured if there is a melanoma dx present elsewhere. This prevents these annotations from being identified as melanoma dx, as some of these types can apply to non-melanoma dx as well
    
    @Language.component("make_hist_type_ent")
    def make_hist_type_ent(doc):
        for ent in doc.ents:
            for mod in ent._.modifiers:
                if mod.category == 'HISTOLOGY_TYPE':
                    start,end = mod.modifier_span
                    new_ent = Span(doc,start,end,label='HISTOLOGY')
                    span = doc[start:end]
                    try:
                        doc.ents += (new_ent,)
                    except:
                        pass
        return doc
        
    nlp.add_pipe("make_hist_type_ent",name="make_hist_type_ent",last=True)
        
    return nlp


def run_nlp(raw_txt,nlp,n_process = 1):
    """
    Runs nlp_pipe on target dataframe column.
    
    Input: String of dataframe column name to run nlp on
    
    Returns: Dataframe column of processed nlp
    """
    if not isinstance(raw_txt,list):
        try:
            raw_txt = raw_txt.tolist()
        except:
            try:
                raw_txt = list(raw_txt)
            except:
                print("Input arg must be type list.")
                return 
    return list(nlp.pipe(raw_txt,n_process=n_process))


def nlp_checker(test_phrase,nlp):
    """Used for quick checks. Given a string, will return the token concept tags, matched entities from target matcher, and the rule used to match."""
    from medspacy.visualization import visualize_dep, visualize_ent
    doc = nlp(test_phrase)
    tok_list = []
    ent_list = []
    for token in doc:
        tok_list.append((token, token._.concept_tag,token._.pretag))
    for ent in doc.ents:
        ent_list.append(str((ent,ent._.target_rule)))
    print(tok_list)
    print()
    print("\n\n".join(ent_list))
    visualize_ent(doc)

# #Helper functions

def map_histology(ent):
    """Helper function to map histology labels using the tags or entity label"""
    if ent.label_ in ["HISTOLOGY","HISTOLOGY_TYPE","TUMOR_HIST"]: #Labels that have specific histologies
        hist_ext = list(set([tok._.pretag.replace(' ','_').replace('\'',"") for tok in ent if tok._.concept_tag in ["HISTOLOGY_TERM"]]))
    elif (ent.label_ in ["MEL_UNSPEC"]):
        hist_ext = ["melanoma_unspecified"]
    elif (ent.label_ in ["TUMOR_UNSPEC","MALIGNANCY"]):
        hist_ext = ["cancer_unspecified"]
    elif (ent.label_ == "PENDING"):
        hist_ext = ["pending"]
    elif ent.label_ in ["NOT_MELANOMA_DX"]:
        hist_ext = ["non_melanoma_dx"]
    else:
        hist_ext = [None] #Everything should be mapped with above conditions
    return hist_ext

def measurement_extract(span):
    """Helper function to extract measurement values for breslow depth"""
    values = []
    units = []
    for tok in span:
        if tok._.concept_tag == 'VALUE':
            values.append(float(tok.text))
        elif tok._.concept_tag == 'UNIT':
            if tok.text.lower() in ['mm','millimeter','millimeters']:
                units.append('mm')
            if tok.text.lower() in ['cm','centimeters','centimeter']:
                units.append('cm')
        elif tok._.concept_tag == 'VALUE_UNIT':
            ext_values = re.findall(r'[\d]*\.?\d+',tok.text)
            ext_units = re.findall(r'cm|mm|millimeters?|centimeters?',tok.text,re.IGNORECASE)
            values.append(float(ext_values[0]))
            if ext_units[0].lower() in ['mm','millimeter','millimeters']:
                units.append('mm')
            if ext_units[0].lower() in ['cm','centimeters','centimeter']:
                units.append('cm')
    values = list(set(values))
    units = list(set(units))
    if (len(units) == 1) and (len(values) == 1):
        if units[0] == 'mm':
            return values[0]
        if units[0] == 'cm':
            return values[0] * 10
    elif (len(units) == 2) and (len(values) == 2): #cases where both the cm and mm are reported (one value must be 10x the other)
        a,b = values
        if (round(a,0) == round(round(b*10,0))) or (round(b,0) == round(a * 10,0)):
            return max(values) #take the larger number as the mm
    else:
        return "EXT_ERROR"

def MI_extract_full(span):
    """Helper function to extract measurement values for mitotic index.
    Returns a list containing [mitotic_mm2_min,mitotic_mm2_max,mitotic_qualitative,mitotic_hpf_numerator_min,mitotic_numerator_max,mitotic_hpf_denominator]"""
    hpf_terms = ['hpf','high','power','fields']
    hpf = 0 #hpf is not default unit (much less common than mm2 and outdated)
    convert_english = {'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9,'ten':10,'eleven':11,'twelve':12,'thirteen':13,'fourteen':14,'fifteen':15}
    num1,num2,qualitative,hpf_num1,hpf_num2,hpf_den = [None,None,None,None,None,None]
    values = []
    tok_tags = [tok._.pretag for tok in span]
    if 'VALUE' in tok_tags:
        for tok in span:
            if tok._.pretag == 'VALUE':
                if tok.text.lower() in (convert_english.keys()):
                    value_text = convert_english[tok.text.lower()]
                else:
                    value_text = tok.text
                try:
                    value_text = str(float(value_text)) #Pointless now
                    values.append(value_text)
                except:
                    print(f"Couldn't cast {value_text} as string")
    if 'RESULT' in tok_tags:
        qualitative = " ".join([tok.text for tok in span if tok._.pretag == 'RESULT'])
        #if result.lower() in ['none','no','not','absent']:
        #    return 'No_MI'
        #else:
        #    return result
    if 'UNIT' in tok_tags:
        if any(term in span.text for term in hpf_terms):
            hpf = True
    if len(values) == 1:
        print('nothing')#something
    return [num1,num2,qualitative,hpf_num1,hpf_num2,hpf_den]
            
    

def MI_extract(span):
    """Helper function to extract measurement values for mitotic index"""
    convert_english = {'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9,'ten':10,'eleven':11,'twelve':12,'thirteen':13,'fourteen':14,'fifteen':15}
    tok_tags = [tok._.pretag for tok in span]
    if 'VALUE' in tok_tags:
        for tok in span:
            if tok._.pretag == 'VALUE':
                if tok.text.lower() in (convert_english.keys()):
                    value_text = convert_english[tok.text.lower()]
                else:
                    value_text = tok.text
                try:
                    return str(float(value_text)) #Pointless now
                except:
                    return value_text
    elif 'RESULT' in tok_tags:
        result = " ".join([tok.text for tok in span if tok._.pretag == 'RESULT'])
        if result.lower() in ['none','no','not','absent']:
            return 'No_MI'
        else:
            return result
    else:
        return 'NA'

def clark_extract(span):
    roman_numeral = {'i':1,'ii':2,'iii':3,'iv':4,'v':5}
    for tok in span:
        ext_values = re.findall(r'^(i{1,4}|v|iv|[0-5])$',tok.text.lower())
        try:
            return (int(ext_values[0]))
        except:
            try:
                return roman_numeral[ext_values[0]]
            except:
                continue
    print(f"Could not find value for {span.text}")
    return np.nan
        


def map_mod(mod,span,ent_label):
    lab = None
    val = None
    if mod.category in ['DEPTH','MELANOMA_DEPTH']:
        lab = 'breslow_depth_mm'
        try:
            val = measurement_extract(span)
        except Exception as e:
            #value = -1
            value = "EXT_ERROR"
    elif mod.category in ['DEPTH_NO_VALUE','MELANOMA_DEPTH_NO_VALUE',\
                          'T_STAGING_GUIDELINE','UNCERTAIN_NO_MEASUREMENT',\
                          'UNCERTAIN_MEASUREMENT','NOT_APPLICABLE','NOT_MEASURED',\
                          'SUPERFICIAL_BIOPSY','SUPERFICIALLY_INVASIVE','NEGATED_METASTATIC','PERIPHERAL_TRANSECTION']:#,'ULCERATION_TERM']: #Unsure about uncertain/not-applicable/non-measured breslow
        lab = 'IGNORE'
        val = None
    elif mod.category in ['GREATER_THAN_EXCISION']:
        lab = 'deep_margin_involvement'
        val = 1
    elif mod.category in ['CLARK_DEPTH']:
        lab = 'clark_level'
        val = clark_extract(span) #replace with clark ext logic
    elif mod.category in ['ULCERATION_ABSENT']:
        lab = 'ulceration_status'
        val = 'absent'
    elif mod.category in ['ULCERATION_IMPENDING']:
        lab = 'ulceration_status'
        val = 'impending'
    elif mod.category in ['ULCERATION_INDETERMINATE']:
        lab = 'ulceration_status'
        val = 'indeterminate'
    elif mod.category in ['ULCERATION_POSITIVE']:
        lab = 'ulceration_status'
        val = 'present'
    elif mod.category in ['ULCERATION_TERM']: ###### Need to make sure context is set up for this then extract negation
        if span.text.strip() != '':
            lab = 'ulceration_status'
            val = 'present'
        else:
            lab = 'IGNORE'
            val = None
    elif mod.category in ['METASTATIC']:
        lab = 'metastasis'
        val = 1
    elif mod.category in ['SKIN_TOPOGRAPHY']:
        lab = 'Skin_Topography'
        val = span.text
    elif mod.category in ['NONSKIN_TOPOGRAPHY']:
        lab = 'Nonskin_Topography'
        val = span.text
    elif mod.category in ['HISTORICAL']:
        lab = 'is_historical'
        val = 1
    elif mod.category in ['HYPOTHETICAL']:
        lab = 'is_hypothetical'
        val = 1
    elif mod.category in ['NEGATED_EXISTENCE']:
        lab = 'is_negated'
        val = 1
    elif mod.category in ['POSSIBLE_EXISTENCE']:
        lab = 'is_possible_existence'
        val = 1
    elif mod.category in ['DEEP_TRANSECTION']:
        if ent_label == 'NOT_MELANOMA_DX':
            lab = 'non_melanoma_transection'
        else:
            lab = 'deep_margin_involvement'
        val = 1
    elif mod.category in ['UNSPEC_TRANSECTION']:
        if ent_label == 'NOT_MELANOMA_DX':
            lab = 'non_melanoma_transection'
        else:
            lab = 'unspecific_or_uncertain_margin_involvement'
        val = 1
    elif mod.category in ['NEGATED_TRANSECTION']:
        lab = 'deep_margin_involvement_negated'
        val = 1
    elif mod.category in ['MITOTIC','MITOTIC_RESULT','DEFER','MITOTIC_DEFER']:
        lab = 'Mitotic_index'
        val = MI_extract(span)
    return (lab,val)

def transect_depth_ext(span):
    tok_list = [tok._.concept_tag for tok in span]
    return (('GREATER' in tok_list[8:]) & ('T_STAGE' not in tok_list))

columns_top_grouped = ['doc_id', 'Topography','Topography_start_span', 'Topography_end_span', 'skin_topography_present','breslow_depth_mm', 'clark_level', 'metastasis','metastasis_historical', 'metastasis_negated','metastasis_hypothetical', 'metastasis_is_possible_existence','nonskin_melanoma_dx', 'deep_margin_involvement', 'unspecific_or_uncertain_margin_involvement','deep_margin_involvement_negated', 'non_melanoma_transection','depth_greater_than_breslow_measurement', 'ulceration_status', 'Mitotic_index','pending', 'cutaneous', 'epithelioid', 'spitzoid', 'acral','amelanotic', 'choroidal', 'mucosal', 'hutchinsons_melanotic_freckle','balloon_cell', 'melanoma_unspecified', 'letiginous', 'lentigo_maligna','in_situ', 'nodular', 'superficial_spreading', 'spindle_cell','melanoma_not_otherwise_specified', 'nevoid', 'ocular', 'desmoplastic','melanoma_negated', 'melanoma_hypothetical','melanoma_is_possible_existence', 'melanoma_historical','cancer_negated', 'non_melanoma_dx', 'cancer_unspecified','non_melanoma_dx_negated', 'cancer_unspecified_negated','non_melanoma_dx_historical', 'cancer_unspecified_historical','non_melanoma_dx_hypothetical', 'cancer_unspecified_hypothetical','non_melanoma_dx_is_possible_existence','cancer_unspecified_is_possible_existence']

empty_df = pd.DataFrame(columns=columns_top_grouped)




# #Data transformation functions

def data_transformation(docIDs,docs):
    long_df = spacy_to_df_melanoma(docIDs,docs)
    if not long_df.empty:
        piv_1 = flatten_on_relationships(long_df)
        piv_1_encoded = one_hot_encode_hist(piv_1)
        top_grouped = grouping_top(piv_1_encoded)
    else:
        piv_1 = None
        piv_1_encoded = None
        long_df = None
        top_grouped = empty_df
    doc_list = top_grouped.doc_id.tolist()
    for doc_id in docIDs:
        if doc_id not in doc_list:
            top_grouped = pd.concat([pd.DataFrame([[doc_id,None ,None ,None ,0 ,None ,None ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,None ,None ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]],columns=columns_top_grouped),top_grouped])
            #top_grouped = pd.concat([pd.DataFrame([[doc_id ,None ,None ,None ,None ,None ,None ,None ,None ,None ,None ,None ,None ,None ,None ,None ,None ,None ,None ,None ,None ,None ,None ,None ,None ,None ,None ,None ,None ,None ,None ,None ,None ,None ,None ,None ,None ,None ,None ,None ,None ,None ,None ,None ,None ,None ,None ,None ,None ,None ,None ,None ,None ,None ,None ,None ]],columns=columns_top_grouped),top_grouped])
    
    return {"long_df":long_df,"piv_1":piv_1,"piv_1_encoded":piv_1_encoded,"top_grouped":top_grouped}


def spacy_to_df_melanoma(docIDs,docs):
    long_data = []
    for docID,doc in zip(docIDs,docs):
        for ent in doc.ents:
            hist_ext = map_histology(ent)
            if len(ent._.modifiers) > 0:
                for mod in ent._.modifiers:
                    span = doc[mod.modifier_span[0]:mod.modifier_span[1]]
                    mod_ext_label,mod_ext_value = map_mod(mod,span,ent.label_)
                    try:
                        mod_end_char = doc[mod.modifier_span[1]].idx #throws error if index is at the end of doc
                    except:
                        mod_end_char = doc[mod.modifier_span[1]-1].idx
                    long_data.append({
                            'doc_id':docID,
                            'anchor_text':ent.text,
                            'anchor_label':ent.label_,
                            'anchor_start_char':ent.start_char,
                            'anchor_end_char':ent.end_char,
                            'anchor_mapping':hist_ext,
                            'modifier_text': span.text,
                            'modifier_category': mod.category,
                            'modifier_start_char':doc[mod.modifier_span[0]].idx,
                            'modifier_end_char':mod_end_char,
                            'modifier_ext_label':mod_ext_label,
                            'modifier_ext_value':mod_ext_value
                    })
                    if mod.category in ['DEPTH','MELANOMA_DEPTH','CLARK_DEPTH']:
                        try:
                            span = doc[mod.modifier_span[0]-12:mod.modifier_span[1]+4]
                        except:
                            span = doc[mod.modifier_span[0]:mod.modifier_span[1]]
                        mod_ext_label,mod_ext_value = 'depth_greater_than_breslow_measurement',transect_depth_ext(span)
                        long_data.append({
                            'doc_id':docID,
                            'anchor_text':ent.text,
                            'anchor_label':ent.label_,
                            'anchor_start_char':ent.start_char,
                            'anchor_end_char':ent.end_char,
                            'anchor_mapping':hist_ext,
                            'modifier_text': span.text,
                            'modifier_category': mod.category,
                            'modifier_start_char':doc[mod.modifier_span[0]].idx,
                            'modifier_end_char':mod_end_char,
                            'modifier_ext_label':mod_ext_label,
                            'modifier_ext_value':mod_ext_value
                        })
            else:
                long_data.append({
                            'doc_id':docID,
                            'anchor_text':ent.text,
                            'anchor_label':ent.label_,
                            'anchor_start_char':ent.start_char,
                            'anchor_end_char':ent.end_char,
                            'anchor_mapping':hist_ext,
                            'modifier_text': None,
                            'modifier_category': None,
                            'modifier_start_char':None,
                            'modifier_end_char':None,
                            'modifier_ext_label':'IGNORE',
                            'modifier_ext_value':'None'
                    })
    if len(long_data) == 0:
        return empty_df
    long_df = pd.DataFrame(long_data).explode('anchor_mapping').reset_index(drop=True)
    long_df.loc[(long_df['anchor_mapping'] == 'in_situ') & (long_df['modifier_ext_label'] == 'unspecific_or_uncertain_margin_involvement'),'modifier_ext_label'] = 'PERIPHERAL_TRANSECTION'
    return long_df

def flatten_on_relationships(long_df):
    long_df['modifier_ext_value'] = long_df.apply(lambda row: f"{row['modifier_ext_value']}_{row['modifier_start_char']}_{row['modifier_end_char']}" if row['modifier_ext_label'] in ['Skin_Topography','Nonskin_Topography'] else row['modifier_ext_value'],axis=1)
    piv_1 = long_df.pivot_table(index=['doc_id','anchor_text','anchor_mapping','anchor_start_char','anchor_end_char'],columns='modifier_ext_label',values='modifier_ext_value',aggfunc=list)
    def first_return(x):
        try:
            return x[0]
        except:
            return None
    try:
        piv_1.Skin_Topography = piv_1.Skin_Topography.apply(first_return)
    except:
        piv_1["Skin_Topography"] = None
    try:
        piv_1.Nonskin_Topography = piv_1.Nonskin_Topography.apply(first_return)
    except:
        piv_1['Nonskin_Topography'] = None
    for col in ['is_negated','is_historical','is_hypothetical','is_possible_existence','metastasis']:
        if col not in piv_1.columns:
            print("Either no negation, historical, possible existence, or hypothetical contexts detected in this set. Setting negation/history/hypothetical to None")
            piv_1[col] = None
    piv_1.is_negated = piv_1.is_negated.apply(first_return)
    piv_1.is_historical = piv_1.is_historical.apply(first_return)
    piv_1.is_hypothetical = piv_1.is_hypothetical.apply(first_return)
    piv_1.is_possible_existence = piv_1.is_possible_existence.apply(first_return)
    #piv_1.breslow_depth_mm = piv_1.breslow_depth_mm.apply(first_return)
    return piv_1.reset_index()

def replace_with_one(mixed_list):
    if not isinstance(mixed_list,list):
        mixed_list = [mixed_list]
    flat_list = [item for sublist in mixed_list for item in (sublist if isinstance(sublist,list) else [sublist])]
    return 1 if 1 in flat_list else 0

cols_in_df = ['breslow_depth_mm',
       'clark_level', 'metastasis','ulceration_status','pending','deep_margin_involvement','unspecific_or_uncertain_margin_involvement','deep_margin_involvement_negated','non_melanoma_transection','depth_greater_than_breslow_measurement','Mitotic_index']
melanoma_dx =['epithelioid','spitzoid','acral','amelanotic','choroidal','mucosal','hutchinsons_melanotic_freckle','balloon_cell','melanoma_unspecified','letiginous','lentigo_maligna','in_situ','nodular','superficial_spreading','spindle_cell','melanoma_not_otherwise_specified','nevoid','ocular','desmoplastic']

other_dx = ['non_melanoma_dx','cancer_unspecified']

def one_hot_encode_hist(piv_1,exclude_nonmelanoma_ulceration = 1):
    #if the cancer is negated or historical, capture this separately
    piv_1['is_negated'] = piv_1['is_negated'].fillna(0)
    piv_1['anchor_mapping'] = piv_1.apply(lambda x: f"{x['anchor_mapping']}_historical" if ((x['is_historical'] == 1) & (x['is_negated'] == 0)) else x["anchor_mapping"], axis=1)
    piv_1['anchor_mapping'] = piv_1.apply(lambda x: f"{x['anchor_mapping']}_negated" if x['is_negated'] == 1 else x["anchor_mapping"], axis=1)
    piv_1['anchor_mapping'] = piv_1.apply(lambda x: f"{x['anchor_mapping']}_hypothetical" if ((x['is_hypothetical'] == 1) & (x['is_negated'] == 0)) else x["anchor_mapping"], axis=1) #is_possible_existence
    piv_1['anchor_mapping'] = piv_1.apply(lambda x: f"{x['anchor_mapping']}_is_possible_existence" if ((x['is_possible_existence'] == 1) & (x['is_negated'] == 0)) else x["anchor_mapping"], axis=1)
    #If the metastasis is connected to a negated/historical cancer, capture this separately
    piv_1['metastasis'] = piv_1['metastasis'].apply(lambda x: [x])
    piv_1['metastasis'] = piv_1['metastasis'].apply(replace_with_one)
    piv_1['metastasis_historical'] = piv_1.apply(lambda x: 1 if ((x['is_historical'] == 1) & (x['is_negated'] == 0) & (x["metastasis"] == 1)) else 0, axis=1)
    piv_1['metastasis_negated'] = piv_1.apply(lambda x: 1 if ((x['is_negated'] == 1) & (x["metastasis"] == 1)) else 0, axis=1)
    piv_1['metastasis_hypothetical'] = piv_1.apply(lambda x: 1 if ((x['is_hypothetical'] == 1) & (x['is_negated'] == 0) & (x["metastasis"] == 1)) else 0, axis=1)
    piv_1['metastasis_is_possible_existence'] = piv_1.apply(lambda x: 1 if ((x['is_possible_existence'] == 1) & (x['is_negated'] == 0) & (x["metastasis"] == 1)) else 0, axis=1)
    piv_1['metastasis'] = piv_1.apply(lambda x: 0 if ((x["metastasis_historical"] == 1) | (x["metastasis_negated"] == 1) | (x["metastasis_hypothetical"] == 1) | (x["metastasis_is_possible_existence"] == 1)) else x["metastasis"], axis=1) #metastasis should be 0 if it's historical or negated or hypothetical (this is captured separately)
    one_hot_encoded_hist = pd.get_dummies(piv_1['anchor_mapping'])
    piv_1_encoded = pd.concat([piv_1,one_hot_encoded_hist],axis=1)
    piv_1_encoded['skin_topography_present'] = piv_1_encoded.Skin_Topography.apply(lambda x: 1 if pd.notna(x) else 0)
    piv_1_encoded['Topography'] = piv_1_encoded.apply(lambda row: row['Skin_Topography'] if pd.notna(row['Skin_Topography']) else row['Nonskin_Topography'],axis=1)
    piv_1_encoded['Topography_start_span'] = piv_1_encoded.Topography.apply(lambda x: x.split("_")[-2] if pd.notna(x) else None)
    piv_1_encoded['Topography_end_span'] = piv_1_encoded.Topography.apply(lambda x: x.split("_")[-1] if pd.notna(x) else None)
    piv_1_encoded['Topography'] = piv_1_encoded.Topography.apply(lambda x: "".join(x.split("_")[:-2]) if pd.notna(x) else None)
    #piv_1_encoded = piv_1_encoded[piv_1_encoded.is_negated != 1]
    
    for col in cols_in_df + melanoma_dx + other_dx + [f"{x}_negated" for x in melanoma_dx] + [f"{x}_historical" for x in melanoma_dx] + [f"{x}_negated" for x in other_dx] + [f"{x}_historical" for x in other_dx] + [f"{x}_hypothetical" for x in melanoma_dx] + [f"{x}_hypothetical" for x in other_dx] + [f"{x}_is_possible_existence" for x in melanoma_dx] + [f"{x}_is_possible_existence" for x in other_dx]:
        if col not in piv_1_encoded.columns:
            piv_1_encoded[col] = [[None] for _ in range(piv_1_encoded.shape[0])]

    if exclude_nonmelanoma_ulceration:
        piv_1_encoded.loc[piv_1_encoded['anchor_mapping'].isin(['non_melanoma_dx']),'ulceration_status'] = None
    return piv_1_encoded

melanoma_cutaneous = ['desmoplastic','epithelioid','lentigo_maligna','nevoid','nodular','melanoma_not_otherwise_specified','superficial_spreading','spindle_cell','spitzoid',
         'letiginous',
        'acral',
         'amelanotic',
         'hutchinsons_melanotic_freckle',
         'balloon_cell'] #'melanoma_unspecified' must also have no in-situ in order to count

melanoma_ocular = ['choroidal','ocular']

melanoma_in_situ = ['in_situ']

melanoma_mucosal = ['mucosal']

def grouping_top(piv_1_encoded,return_max_values=1):
    piv_1_encoded[['Topography',"Topography_start_span","Topography_end_span"]] = piv_1_encoded[['Topography',"Topography_start_span","Topography_end_span"]].fillna('None')
    top_grouped = piv_1_encoded.groupby(['doc_id','Topography',"Topography_start_span","Topography_end_span"]).agg(list)
    
    def ulceration_priority(mixed_list):
        if not isinstance(mixed_list,list):
            mixed_list = [mixed_list]
        flat_list = [item for sublist in mixed_list for item in (sublist if isinstance(sublist,list) else [sublist])]
        priority = ['present','impending','absent','indeterminate']
        for priority_val in priority:
            if priority_val in flat_list:
                return priority_val
        return None
    
    def max_breslow(mixed_list):
        if not isinstance(mixed_list,list):
            mixed_list = [mixed_list]
        flat_list = [item for sublist in mixed_list for item in (sublist if isinstance(sublist,list) else [sublist])]
        flat_list = [num for num in flat_list if num not in [None,'EXT_ERROR'] and pd.notna(num)]
        if len(flat_list) > 0:
            if return_max_values:
                return max(flat_list)
            else:
                return set(flat_list)
        else:
            return None
    
    def clark_level_sum(mixed_list):
        if not isinstance(mixed_list,list):
            mixed_list = [mixed_list]
        flat_list = [item for sublist in mixed_list for item in (sublist if isinstance(sublist,list) else [sublist])]
        flat_list = list(set([str(el) for el in flat_list if el not in [None] and isinstance(el,str)]))
        if len(flat_list) > 0:
            return "; ".join(flat_list)
        else:
            return None

    def max_Mitotic_index(mixed_list):
        if not isinstance(mixed_list,list):
            mixed_list = [mixed_list]
        flat_list = [item for sublist in mixed_list for item in (sublist if isinstance(sublist,list) else [sublist])]
        flat_list = [num for num in flat_list if num not in [None,'NA'] and pd.notna(num)]
        flat_list_num = []
        for el in flat_list:
            if el != None:
                try:
                    flat_list_num.append(float(el))
                except:
                    pass
        if len(flat_list_num) > 0:
            if return_max_values:
                return str(max(flat_list_num))
            else:
                return set(flat_list_num)
        elif len(flat_list) > 0:
            return flat_list[0] #arbitrary
        else:
            return None
    
    top_grouped = top_grouped.reset_index()
    columns_to_adjust = ["metastasis","metastasis_historical","metastasis_negated","metastasis_hypothetical","metastasis_is_possible_existence","skin_topography_present","pending",'deep_margin_involvement','unspecific_or_uncertain_margin_involvement','deep_margin_involvement_negated','non_melanoma_transection','depth_greater_than_breslow_measurement'] + melanoma_dx + [f"{x}_negated" for x in melanoma_dx] + [f"{x}_historical" for x in melanoma_dx] + other_dx + [f"{x}_negated" for x in other_dx] + [f"{x}_historical" for x in other_dx] + [f"{x}_hypothetical" for x in melanoma_dx] + [f"{x}_hypothetical" for x in other_dx] + [f"{x}_is_possible_existence" for x in melanoma_dx] + [f"{x}_is_possible_existence" for x in other_dx]
    #top_grouped[columns_to_adjust] = top_grouped[columns_to_adjust].applymap(replace_with_one)
    #top_grouped[columns_to_adjust] = top_grouped[columns_to_adjust].apply(lambda col: col.apply(replace_with_one))
    for col in columns_to_adjust:
        top_grouped[col] = top_grouped[col].apply(replace_with_one)
    top_grouped['breslow_depth_mm'] = top_grouped['breslow_depth_mm'].apply(max_breslow)
    top_grouped['ulceration_status'] = top_grouped['ulceration_status'].apply(ulceration_priority)
    top_grouped['clark_level'] = top_grouped['clark_level'].apply(max_breslow)
    top_grouped['melanoma_negated'] = top_grouped[[f"{x}_negated" for x in melanoma_dx]].any(axis=1).astype(int)
    top_grouped['melanoma_hypothetical'] = top_grouped[[f"{x}_hypothetical" for x in melanoma_dx]].any(axis=1).astype(int)
    top_grouped['melanoma_is_possible_existence'] = top_grouped[[f"{x}_is_possible_existence" for x in melanoma_dx]].any(axis=1).astype(int)
    top_grouped['melanoma_historical'] = top_grouped[[f"{x}_historical" for x in melanoma_dx]].any(axis=1).astype(int)
    top_grouped['Mitotic_index'] = top_grouped['Mitotic_index'].apply(max_Mitotic_index)
    top_grouped.loc[top_grouped.Topography.str.contains('eye',case=False) & (~top_grouped.Topography.str.contains('eyelid',case=False)) & (top_grouped.melanoma_unspecified == 1),'ocular'] = 1 #eye topographies are ocular melanoma
    top_grouped['cutaneous'] = top_grouped[melanoma_cutaneous].any(axis=1).astype(int)

    ##Setting melanoma unsp to only work with not also in situ
    top_grouped.loc[(top_grouped['in_situ'] != 1) & (top_grouped['melanoma_unspecified'] == 1),'cutaneous'] = 1
    
    top_grouped['ocular'] = top_grouped[melanoma_ocular].any(axis=1).astype(int)
    top_grouped['in_situ'] = top_grouped[melanoma_in_situ].any(axis=1).astype(int)
    top_grouped['mucosal'] = top_grouped[melanoma_mucosal].any(axis=1).astype(int)
    top_grouped['nonskin_melanoma_dx'] = 0 #default to 0
    top_grouped.loc[(top_grouped['melanoma_unspecified'] == 1) & (top_grouped["Topography"].notna() & (top_grouped["Topography"] != 'None')) & (top_grouped['skin_topography_present'] == 0), 'nonskin_melanoma_dx'] = 1
    top_grouped['cancer_negated'] = top_grouped[['melanoma_negated','cancer_unspecified_negated']].any(axis=1).astype(int)
    #top_grouped[['ReportID','DocumentID','SourceTable']] = top_grouped['doc_id'].str.split('_',expand=True)
    return top_grouped[['doc_id','Topography', 'Topography_start_span', 'Topography_end_span','skin_topography_present', 'breslow_depth_mm','clark_level', 'metastasis',"metastasis_historical","metastasis_negated","metastasis_hypothetical","metastasis_is_possible_existence",'nonskin_melanoma_dx','deep_margin_involvement','unspecific_or_uncertain_margin_involvement','deep_margin_involvement_negated','non_melanoma_transection','depth_greater_than_breslow_measurement','ulceration_status','Mitotic_index','pending','cutaneous']+melanoma_dx+['melanoma_negated','melanoma_hypothetical','melanoma_is_possible_existence','melanoma_historical','cancer_negated']+other_dx+[f"{x}_negated" for x in other_dx] + [f"{x}_historical" for x in other_dx] + [f"{x}_hypothetical" for x in other_dx] + [f"{x}_is_possible_existence" for x in other_dx]]

# [['doc_id', 'Topography', 'Topography_start_span', 'Topography_end_span','skin_topography_present', 'breslow_depth_mm','clark_level', 'metastasis','ulceration_status', 'cancer_unspecified', 'desmoplastic','epithelioid', 'in_situ', 'lentigo_maligna', 'melanoma_unspecified','nevoid', 'nodular', 'spitzoid','letiginous','acral','amelanotic','choroidal','mucosal','ocular','hutchinsons_melanotic_freckle','balloon_cell', 'non_melanoma_dx', 'melanoma_not_otherwise_specified','spindle_cell', 'superficial_spreading','pending']]

def doc_level_mel(top_grouped):
    return top_grouped.groupby('doc_id').agg({'cutaneous':'max','ocular':'max','in_situ':'max','mucosal':'max','nonskin_melanoma_dx':'max','metastasis':'max',"metastasis_negated":'max','cancer_negated':'max'}).reset_index()



def upload_to_cdw(df,dest_table,db_name,db_server: str, db_db: str,annotated_span_len = 5000,varchar_len = 400,other_int_col = [],conn_list=[],other_float_col=[]):
    """Upload df to cdw in one go
        Will assume that DocID/docName are bigint, spanStartChar/spanEndChar are int,
        annotatedSpan is varchar 2000, and everything
        else is varchar 400, unless otherwise specified"""
    import pyodbc
    
    print(f'Closing {str(len(conn_list))} connections')
    for con in conn_list:
        con.close()
    conn_list = []
    db_pwd = ''

    #Set cursor
    conn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server}'
                          +';SERVER='+db_server
                          +';DATABASE='+db_db
                          +';TRUSTED_CONNECTION=yes')
    cursor = conn.cursor()
    
    conn_list.append(conn)
    try:
        import regex as re
        column_names = [re.sub(r'^[\d_]+','',cols) for cols in df.columns]
        
        create_query = f'''drop table if exists {dest_table};CREATE TABLE {dest_table}(
        '''
        for col in column_names:
            if (col == 'docID') or (col == 'docName') or (col == 'TIUDocumentSID'):
                #create_query += f'[{col}] [bigint] NULL,\n'
                create_query += f'[{col}] [varchar] ({varchar_len}) NULL,\n'
            elif (col == 'spanStartChar') or (col == 'spanEndChar') or (col == 'start_loc') or (col == 'end_loc'):
                create_query += f'[{col}] [int] NULL,\n'     
            elif ('annotatedSpan' in col):
                create_query += f'[{col}] [varchar] ({annotated_span_len}) NULL,\n'
            elif ('Span_Text' in col):
                create_query += f'[{col}] [varchar] ({annotated_span_len}) NULL,\n'
            elif (col in other_int_col):
                create_query += f'[{col}] [int] NULL,\n'
            elif (col in other_float_col):
                create_query += f'[{col}] [FLOAT] NULL,\n'
            else:
                create_query += f'[{col}] [varchar] ({varchar_len}) NULL,\n'
        create_query += f'''); CREATE CLUSTERED COLUMNSTORE INDEX CCI ON {dest_table}'''

        cursor.execute(create_query)
        print('Table created')

        res = [row.tolist() for row in df.replace('NA',None,inplace=False).fillna('None').values]
        print('\nResultant list of lists created for dataframe. First row of df: ' + str(res[0]))

        output = f'insert into {dest_table} ('
        for col in column_names:
            output += f'{col}, '
        output = output[:-2] + ') values (' #Take off the last comma
        for col in column_names:
            output += f'?, '
        output = output[:-2] + ')'

        print('\nUploading to CDW using: ' + output)
        cursor.executemany(output,res)
        conn.commit()

        cursor.close()
        conn.close()
        
        return []
    except Exception as e:
        print("\n\nError: ",e)
        cursor.close()
        conn.close()
        return conn_list



def transform_annot_df(annot_df,include_no_top_annot=True,binary_classification = True):
    annot_df_adj = annot_df[annot_df['spanLabel'].isin(['1_Cancer_Diagnosis','2_Melanoma_Histology_Dx','3_Histology_all_other_cancers','Melanoma_Breslow_thickness','Melanoma_Clark_level_of_invasion','Melanoma_Ulceration','Mitotic_index','Metastasis','Margins'])]
    annot_df_adj = annot_df_adj[annot_df_adj['Melanoma_Breslow_thickness_Presence'] != 'Unable to determine or missing']
    annot_df_adj = annot_df_adj[~((annot_df_adj['spanLabel'] == 'Metastasis') & (annot_df_adj['Metastasis_Site_Num1_of_metastasis'].isin(['None'])))]
    annot_df_adj = annot_df_adj.loc[:,(annot_df_adj != 'None').any(axis=0)]
    
    if include_no_top_annot:
        ghost_dx_dict = {}
        for col in annot_df_adj.columns:
            ghost_dx_dict[col] = []
        for i,textid in enumerate(annot_df_adj.textID.unique()):
            for col in annot_df_adj.columns:
                if col in ['Cancer_Diagnosis_Status','Cancer_Diagnosis_Result','Cancer_Diagnosis_Temporality']:
                    ghost_dx_dict[col].append("GHOST_MALIGNANT")
                elif col in ['textID']:
                    ghost_dx_dict[col].append(textid)
                elif col in ['spanLabel']:
                    ghost_dx_dict[col].append('1_Cancer_Diagnosis')
                elif col in ['spanID']:
                    ghost_dx_dict[col].append(textid+"_"+str(i))
                else:
                    ghost_dx_dict[col].append("None")
        ghost_dx = pd.DataFrame.from_dict(ghost_dx_dict)
        annot_df_adj = pd.concat([annot_df_adj,ghost_dx])
        for textid in annot_df_adj.textID.unique():
            annot_df_adj.loc[(annot_df_adj['textID'] == textid) & ((annot_df_adj['relID'] == "None") | ((~annot_df_adj['relID'].isin(annot_df_adj.loc[annot_df_adj['textID'] == textid,'spanID'])) & (annot_df_adj['relID'] != 'None'))),"relID"] = annot_df_adj[(annot_df_adj['textID'] == textid) & (annot_df_adj['spanLabel'] == '1_Cancer_Diagnosis')]['spanID'].tolist()[0]
            #annot_df.loc[(annot_df['textID'] == textid) & ((annot_df['relID'] == "None") | ((~annot_df['relID'].isin(annot_df['spanID'])) & (annot_df['relID'] != 'None'))),"relID"] = annot_df_adj[(annot_df_adj['textID'] == textid) & (annot_df_adj['spanLabel'] == '1_Cancer_Diagnosis')]['spanID'].tolist()[0]


    mapping_hist = {'None':0,
                    'Other melanoma or NOS': 1,
                    'Superficial spreading melanoma': 1,
                    'Nevoid melanoma':1,
                    'Melanoma in situ': 0,
                    'Residual cancer noted':1, #Might want this to be 0
                   'Lentigo maligna melanoma': 1,
                    'Desmoplastic melanoma':1,
                   'Nodular melanoma':1,
                    'Needs further review': 1,
                    'Ocular melanoma': 1,
                    'Unable to determine/needs further review':1,
                    'Spitzoid melanoma':1,
                    'Mucosal melanoma':1,
                   'Acral letiginous melanoma':1,
                   'Melanoma NOS':1}

    mapping_hist_nonmel = {'None':0,
                           'Basal cell (BCC)':1,
                           'Squamous cell (SCC)': 1,
           'Residual cancer noted': 1, 
            'Lymphoma': 1,
            'Unable to determine': 0,
           'Sarcoma': 1,
                          'Other':1,
                          'Carcinoma_Other':1}

    mapping_clark = {'None':0,
                     'Level 2':2, 'Level 3':3, 'Level 4':4, 'Level 5':5, 'Level 1':1,
           'Unable to determine or missing':0}

    mapping_ulcer = {'None':0, 'Absent':0, 'Yes, present':1, 'Unable to determine or missing':0}

    mapping_mitotic = {'None':0, 'Measurement':1, 'Other':0, 'Unable to determine or missing':0}

    mapping_met = {'None':0, 'Not present':0, 'Yes present':1,
           'Unable to determine or missing':0}


    ###Copy breslow annotations, rename labels as depth transection annotations

    annot_df_adj_breslow_transect = annot_df_adj[annot_df_adj['spanLabel'] == 'Melanoma_Breslow_thickness'].copy()
    annot_df_adj_breslow_transect.loc[:,'spanLabel'] = 'Breslow_transect'
    annot_df_adj = pd.concat([annot_df_adj,annot_df_adj_breslow_transect])
    
    mod_ext_list = []
    mod_ext_lab = []
    for i, row in annot_df_adj.iterrows():
        lab = row['spanLabel']
        if lab == '2_Melanoma_Histology_Dx':
            hist = row['Melanoma_Histology_Dx_Type']
            mod_ext_lab.append('Melanoma')
            mod_ext_list.append(mapping_hist[hist])
            if 'situ' in row['annotatedSpan'].lower():
                mod_ext_list[-1] = 0 #Cases that are marked residual but are in situ
        if lab == '3_Histology_all_other_cancers':
            hist = row['Histology_all_other_cancers_Type']
            mod_ext_lab.append('Non_Melanoma')
            mod_ext_list.append(mapping_hist_nonmel[hist])
        if lab == 'Melanoma_Breslow_thickness':
            mod_ext_lab.append('breslow_measurement') #Already filtered out irrelevant cases in above cell
            if binary_classification:
                mod_ext_list.append(1 if (not pd.isna(row['breslow_measurement'])) | (row['Melanoma_Breslow_thickness_Presence'] == 'Measurement') else 0)
            else:
                mod_ext_list.append(row['breslow_measurement'])
        if lab == 'Melanoma_Clark_level_of_invasion':
            clark_level = row['Melanoma_Clark_level_of_invasion_Presence']
            mod_ext_lab.append('clark')
            if binary_classification:
                mod_ext_list.append(1 if mapping_clark[clark_level] > 0 else 0)
            else:
                if mapping_clark[clark_level] == 0:
                    mod_ext_list.append(np.nan)
                else:
                    mod_ext_list.append(mapping_clark[clark_level])
        if lab == 'Melanoma_Ulceration':
            ulceration_stat = row['Melanoma_Ulceration_Presence']
            mod_ext_lab.append('ulceration')
            mod_ext_list.append(mapping_ulcer[ulceration_stat])
        if lab == 'Mitotic_index':
            #if ' 0' in row['annotatedSpan'] or 'None' in row['annotatedSpan']:
            #    mod_ext_lab.append('mitotic')
            #    mod_ext_list.append(0)
            #else:
            mitotic_ind = row['Mitotic_index_Number_of_mitoses_per_mm2_tumor']
            mod_ext_lab.append('mitotic')
            if binary_classification:
                if mitotic_ind == 'Measurement':
                    mod_ext_list.append(1)
                else:
                    try:
                        mod_ext_list.append(1 if (float(row['mitotic_measurement']) > 0) else 0)
                    except:
                        mod_ext_list.append(0)
            else:
                mod_ext_list.append(row['mitotic_measurement'])
        if lab == 'Metastasis':
            met_ind = row['Metastasis_Presence']
            mod_ext_lab.append('metastasis')
            mod_ext_list.append(mapping_met[met_ind])
        if lab == '1_Cancer_Diagnosis':
            #result not 'Negative for residual cancer','Benign-NOT cancer')
            #status not negation probable
            mod_ext_lab.append('Cancer_dx')
            if (row['Cancer_Diagnosis_Status'] not in ['Negation']) and (row['Cancer_Diagnosis_Result'] not in ['Benign-NOT cancer','Unable to determine or missing']) and (row['Cancer_Diagnosis_Temporality'] != 'Past'): #(row['Cancer_Diagnosis_Topography'] == 'Skin')
                mod_ext_list.append(1)
            elif row['Cancer_Diagnosis_Topography'] in ['Lymph node']:
                mod_ext_list.append(2)
            else:
                mod_ext_list.append(0)
        if lab == 'Margins':
            mod_ext_lab.append('transection')
            try:
                mod_ext_list.append(int(row['deep_margin']))
            except:
                mod_ext_list.append(0)
        if lab == 'Breslow_transect':
            mod_ext_lab.append('transection')
            try:
                
                mod_ext_list.append(1 if int(row['depth_greater_than_excision']) == 1 else 0)
            except:
                mod_ext_list.append(0)

    annot_df_adj['modifier_ext_value'] = mod_ext_list
    annot_df_adj['modifier_ext_label'] = mod_ext_lab
    
    annot_piv_1 = annot_df_adj.pivot_table(index=['textID','annotatedSpan','spanStartChar','spanEndChar','spanLabel','relID','spanID'],columns='modifier_ext_label',values='modifier_ext_value',aggfunc='first').reset_index()
    for col in ['Melanoma', 'Non_Melanoma', 'clark','metastasis', 'mitotic', 'breslow_measurement', 'ulceration','transection']:
        if col not in annot_piv_1.columns:
            print(f"No {col} found. Replacing with None")
            annot_piv_1[col] = None
    annot_piv_1_mod = annot_piv_1[annot_piv_1.spanLabel != '1_Cancer_Diagnosis'][['relID','Melanoma', 'Non_Melanoma', 'clark','metastasis', 'mitotic', 'breslow_measurement', 'ulceration','textID','transection']]
    annot_piv_1_dx = annot_piv_1[annot_piv_1.spanLabel == '1_Cancer_Diagnosis'][['textID','annotatedSpan','spanStartChar','spanEndChar','spanLabel','spanID','Cancer_dx']]
    annot_dx_df = annot_piv_1_dx.merge(annot_piv_1_mod,left_on=['textID','spanID'],right_on=['textID','relID'],how='left')
    annot_dx_df[['Melanoma','Non_Melanoma','metastasis','ulceration','transection']] = annot_dx_df[['Melanoma','Non_Melanoma','metastasis','ulceration','transection']].fillna(0).astype(int)
    annot_dx_df[['clark','mitotic','breslow_measurement']] = annot_dx_df[['clark','mitotic','breslow_measurement']].astype(float)
    
    ##Setting using max value per dx
    annot_dx_df = annot_dx_df.groupby(['textID','annotatedSpan','spanStartChar','spanEndChar']).agg(max).reset_index()
    #annot_dx_df.columns = ['textID', 'Topography', 'Topography_start_span','Topography_end_span', 'spanLabel','spanID','Cancer_dx', 'relID', 'Melanoma', 'Non_Melanoma', 'clark', 'metastasis_','mitotic', 'thickness', 'ulceration']
    annot_dx_df.loc[annot_dx_df['Cancer_dx'] == 0,'Melanoma'] = 0
    annot_dx_df.loc[annot_dx_df['Melanoma'] == 0,'transection'] = 0
    
    #setting: ulceration for melanoma only
    #annot_dx_df.loc[annot_dx_df['Melanoma'] != 1,'ulceration'] = 0

    #setting clark only for non-lymph node cases (this is an edge case)
    annot_dx_df.loc[annot_dx_df['Cancer_dx'] == 2,'clark'] = np.nan
    
    annot_dx_df[['Melanoma','Non_Melanoma','metastasis','ulceration','transection']] = annot_dx_df[['Melanoma','Non_Melanoma','metastasis','ulceration','transection']].fillna(0).astype(int)
    annot_dx_df[['clark','mitotic','breslow_measurement']] = annot_dx_df[['clark','mitotic','breslow_measurement']].astype(float)
    
    annot_doc_df = annot_dx_df[['textID','breslow_measurement', 'clark', 'metastasis', 'ulceration', 'Melanoma', 'mitotic', 'Non_Melanoma','transection']].copy()
    
    if binary_classification:
        annot_doc_df = annot_doc_df.groupby('textID').agg(max).reset_index()
    else:
        annot_doc_df = annot_doc_df.groupby('textID').agg({'Melanoma':'max','breslow_measurement':lambda x: set([float(v) for v in x if pd.notna(v)]),'clark':lambda x: set([v for v in x if pd.notna(v)]),'metastasis':'max','ulceration':'max','mitotic':lambda x: set([float(v) for v in x if pd.notna(v)]),'metastasis':'max','transection':'max'}).reset_index()
    
    annot_doc_df[['melanoma','ulceration_status','Mitotic_index']] = annot_doc_df[['Melanoma','ulceration','mitotic']]
    annot_doc_df.drop(columns=['Melanoma','ulceration','mitotic'],inplace=True)
    return annot_doc_df


def transform_nlp_df(top_grouped,binary_classifier=True):
    nlp_df = pd.DataFrame()
    nlp_df[['doc_id','Topography','Topography_start_span','Topography_end_span','breslow_measurement', 'clark','metastasis','ulceration_status','Mitotic_index','cutaneous','mucosal','ocular','melanoma_hypothetical',	'melanoma_is_possible_existence','melanoma_historical','metastasis_hypothetical','metastasis_is_possible_existence','deep_margin_involvement','unspecific_or_uncertain_margin_involvement','deep_margin_involvement_negated','non_melanoma_transection','depth_greater_than_breslow_measurement']] = top_grouped[['doc_id','Topography','Topography_start_span','Topography_end_span','breslow_depth_mm', 'clark_level','metastasis','ulceration_status','Mitotic_index','cutaneous','mucosal','ocular','melanoma_hypothetical',	'melanoma_is_possible_existence','melanoma_historical','metastasis_hypothetical','metastasis_is_possible_existence','deep_margin_involvement','unspecific_or_uncertain_margin_involvement','deep_margin_involvement_negated','non_melanoma_transection','depth_greater_than_breslow_measurement']].copy()
    nlp_df['melanoma'] = 0
    nlp_df.loc[(nlp_df[['cutaneous','mucosal','ocular','melanoma_hypothetical',	'melanoma_is_possible_existence']] == 1).any(axis=1),'melanoma'] = 1

    nlp_df['transection'] = 0
    nlp_df.loc[(nlp_df['deep_margin_involvement'] == 1),'transection'] = 1
    nlp_df.loc[(nlp_df['depth_greater_than_breslow_measurement'] == 1) & (nlp_df['deep_margin_involvement_negated'] != 1),'transection'] = 1
    
    #nlp_df.loc[(nlp_df['unspecific_or_uncertain_margin_involvement'] == 1) & (nlp_df['deep_margin_involvement_negated'] == 0),'transection'] = 1

    #Does not include cases of nonskin mel cases
    nlp_df.loc[nlp_df['metastasis_is_possible_existence'] == 1,'metastasis'] = 1

    #setting: ulceration for melanoma only
    #nlp_df.loc[(nlp_df['ulceration_status'] == 'present'),'ulceration_status'] = 1
    nlp_df.loc[(nlp_df['ulceration_status'] == 'present') & (nlp_df['melanoma'] == 1),'ulceration_status'] = 1
    nlp_df.loc[nlp_df['ulceration_status'] != 1,'ulceration_status'] = 0
    
    nlp_df.replace('None',np.nan,inplace=True)
    if binary_classifier:
        nlp_df.replace('No_MI',1,inplace=True)
    else:
        nlp_df.replace('No_MI',0,inplace=True)
    if binary_classifier:
        nlp_df.loc[nlp_df['breslow_measurement'].notna(),'breslow_measurement'] = 1
        nlp_df.loc[nlp_df['clark'].notna(),'clark'] = 1
        nlp_df.loc[((nlp_df['Mitotic_index'].notna()) & (nlp_df['Mitotic_index'] != 0)),'Mitotic_index'] = 1
        
        nlp_df.loc[nlp_df['breslow_measurement'] != 1,'breslow_measurement'] = 0
        nlp_df.loc[nlp_df['clark'] != 1,'clark'] = 0
        nlp_df.loc[nlp_df['Mitotic_index'] != 1,'Mitotic_index'] = 0

    def safe_to_float(v):
        try:
            x = float(v)
            return 1
        except:
            return 0
            
    def combine_sets(values):
        combined = set()
        for value in values:
            if isinstance(value, set):
                for val in value:
                    if isinstance(val,int) or (isinstance(val,float)):
                        combined.add(val)
        return combined

    if binary_classifier:
        return nlp_df.groupby('doc_id').agg({'melanoma':'max','breslow_measurement':'max','clark':'max','metastasis':'max','ulceration_status':'max','Mitotic_index':'max','metastasis':'max','transection':'max'}).reset_index()
    else:
        return nlp_df.groupby('doc_id').agg({'melanoma':'max','breslow_measurement':combine_sets,'clark':combine_sets,'metastasis':'max','ulceration_status':'max','Mitotic_index':combine_sets,'metastasis':'max','transection':'max'}).reset_index()



#lambda x: set([float(v) for v in x if pd.notna(v)]









