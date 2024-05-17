# Run this cell on the server with enough RAM
# nohup python countwords.py > foo.out 2> foo.err &

import pandas as pd
import numpy as np

print('Loading data... ', end='', flush=True)

df1 = pd.read_pickle('/gpfs01/berens/data/data/pubmed_processed/clean_2024_df')
df2 = pd.read_pickle('/gpfs01/berens/data/data/pubmed_processed/clean_2024_df_daily_updates_v1')

l1 = np.load('/gpfs01/berens/data/data/pubmed_processed/clean_2024_df_labels.npy')
l2 = np.load('/gpfs01/berens/data/data/pubmed_processed/clean_2024_df_daily_updates_v1_labels.npy')

g1 = np.load('/gpfs01/berens/data/data/pubmed_processed/clean_2024_df_gender_first_author.npy', allow_pickle=True)
g2 = np.load('/gpfs01/berens/data/data/pubmed_processed/clean_2024_df_daily_updates_v1_gender_first_author.npy', allow_pickle=True)

gl1 = np.load('/gpfs01/berens/data/data/pubmed_processed/clean_2024_df_gender_last_author.npy', allow_pickle=True)
gl2 = np.load('/gpfs01/berens/data/data/pubmed_processed/clean_2024_df_daily_updates_v1_gender_last_author.npy', allow_pickle=True)

c1 = np.load('/gpfs01/berens/data/data/pubmed_processed/clean_2024_df_countries_first_author.npy')
c2 = np.load('/gpfs01/berens/data/data/pubmed_processed/clean_2024_df_daily_updates_v1_countries_first_author.npy')

print('done', flush=True)
print('Assembling the dataframe...', end='', flush=True)

df1['Label'] = l1
df2['Label'] = l2

df1['Gender'] = g1
df2['Gender'] = g2

df1['Gender_last'] = gl1
df2['Gender_last'] = gl2

df1['Country'] = c1
df2['Country'] = c2

df = pd.concat([df1, df2])
df = df.groupby(['PMID']).last()

print(f' found {len(df)} unique papers...', end='', flush=True)

df['Year'] = df.Date.str.extract('([12]\d\d\d)').values.astype(int)
df = df[(df.Year >= 2010) & (df.Year <= 2024)]

print(f' kept {len(df)} papers. Done.\n', flush=True)

# Titles filter

ind = df.Title.str.contains('Correction:', regex=False)
ind |= df.Title.str.contains('Correction to:', regex=False)
ind |= df.Title.str.contains('Erratum:', regex=False)
ind |= df.Title.str.contains('Erratum to:', regex=False)
ind |= df.Title.str.contains('Corrigendum:', regex=False)
ind |= df.Title.str.contains('Corrigendum to:', regex=False)
ind |= df.Title.str.contains('Retracted:', regex=False)
ind |= df.Title == 'Retraction'

df.loc[ind, 'AbstractText'] = ''

print(f'Removing {np.sum(ind)} abstracts (corrections/errata/retractions/etc).\n')

# Abstracts filter

to_replace = {
    "&ldquo;": {"&ldquo;": '"', "&rdquo;": '"'},
    "&lsquo;": {"&lsquo;": "'", "&rsquo;": "'"},
    "&nbsp;": {"&nbsp;": " "},
    "&shy;": {"&shy;": ""},
    "&mdash;": {"&mdash;": "---"},
    "&ndash;": {"&ndash;": "--"},
    "u2002": {"\\u2002": " "},
    "<p>": {"<p>": "", "</p>": ""},
    "<em>": {"<em>": "", "</em>": ""},
    "This article": {"^This article has been.*": ""},
    "This manuscript": {"^This manuscript has been.*": ""},
    "The above article": {
        "^The above article.*": "",
        ".*The above article, published online.*": "",
    },
    "http": {"^http.*": ""},
    "For complete details": {
          "\s*For complete details on the use and execution of this protocol.*": "",
    },
    "For further information": {
        "For further information please consult linked data\.*": "",
    },
    "Communicated by": {"\s*\(?Communicated by.{0,100}$": ""},
    "Graphical abstract": {"\.\s*Graphical abstract.*": "."},
    "GRAPHICAL ABSTRACT": {"\.\s*GRAPHICAL ABSTRACT.*": "."},
    "VIDEO ABSTRACT": {"\.\s*VIDEO ABSTRACT.*": "."},
    "Video Abstract": {"\s*Video Abstract Available\.*": ""},
    "MINI ABSTRACT": {"\.\s*MINI ABSTRACT.*": "."},
    "ABSTRACT": {
        "^ABSTRACT[:.]?\s*": "",
        "^Abstract ABSTRACT[:.]?\s*": "",
        "^.{0,200} ABSTRACT: ": "",
    },
    "Abstract": {"^Abstract:?\s*": ""},
    "CONSPECTUS": {"CONSPECTUS: ": ""},
    "THIS ARTICLE": {"\s*THIS ARTICLE HAD BEEN MADE AVAILABLE FREE OF CHARGE.*": ""},
    "Copyright ©": {"\s*Copyright ©.*": ""},
    " © ": {
        "\.\s*[^.]*[0-9]\. © [12].*": ".",
        "\. [a-zA-Z]+\.$": ".",
        "\. [ab-zA-Z]+\.$": ".",
        "\. [abc-zA-Z]+\.$": ".",
        "\. [abcd-zA-Z]+\.$": ".",
        "\. Pediatr Pulmonol.$": ".",
        "\. Lasers Surg.$": ".",
        "\s+© .*": "",
    },
    ".© ": {
        "\.\s*[^.]*[0-9]\.© [12].*": ".",
        "\. [a-zA-Z]+\.$": ".",
        "\. [ab-zA-Z]+\.$": ".",
        "\. [abc-zA-Z]+\.$": ".",
        "\. [abcd-zA-Z]+\.$": ".",
        "\. Pediatr Pulmonol.$": ".",
        "\. Lasers Surg.$": ".",
        "\s+\.© .*": "",
    },
    " ©20": {
        "\.\s*[^.]*[0-9]\. ©20.{0,20}$": ".",
        "\s+©20.{0,20}$": "",
        "\.\s*[^.]*[0-9]\. ©20[0-2][0-9] AACRSee.*": ".",
        "\s+©20[0-2][0-9] AACRSee.*": "",        
    },
    "Wiley Periodicals Inc": {
        "\.\s*[^.]*[0-9]\.\s*Published [12][890][0-9][0-9] Wiley Periodicals Inc.*": ".",
        "\. [a-zA-Z]+\.$": ".",
        "\. [ab-zA-Z]+\.$": ".",
        "\. [abc-zA-Z]+\.$": ".",
        "\. [abcd-zA-Z]+\.$": ".",
        "\s*Published [12][890][0-9][0-9] Wiley Periodicals Inc.*": ".",
    },
    "doi": {
        "\s*doi:\s*10\.[0-9a-zA-Z\.\/\-]*\s*$": "",
        "\s*doi:\s*10\.[0-9a-zA-Z\.\/\-]*\s*\(.*\)\.?\s*$": "",
        "\s*http://dx\.doi\.org/10\.[0-9a-zA-Z\.\/\-]*\s*$": "",
    },
    "DOI": {"\s*DOI: http://dx.doi.org/[0-9a-zA-Z\.\/\-]*\s*$": ""},
    "PMID": {".*PubMed PMID: [0-9]*\.\s*": ""},
    "Epub": {"\sEpub.{0,10}[12][890][0-9][0-9]\.?\s*$": ""},
    "Level of Evidence": {"\s*Level of Evidence:?\s*[0-9IV].*": ""},    
    "LEVEL OF EVIDENCE": {"\s*LEVEL OF EVIDENCE:?\s*[0-9IV].*": ""},    
    "Technical Efficacy": {"\s*[0-9] Technical Efficacy: Stage [0-9].*": ""},
    "Geriatr": {
        "\s*Geriatr Gerontol Int [\s0-9,;:\-\.\(\)]*$": "",
        "\s*J Am Geriatr Soc [\s0-9,;:\-\.\(\)]*$": "",
    },
    "Genet Med": {"\s*Genet Med [\s0-9,;:\-\.\(\)]*$": ""},
    "Ann Neurol": {
        # Sometimes occurs twice
        "\s*Ann Neurol [\s0-9,;:\-\.\(\)]*$": "",
        "\s*Ann Neurol [\s01-9,;:\-\.\(\)]*$": "",
    },
    "ANN NEUROL": {"\s*ANN NEUROL [\s0-9,;:\-\.\(\)]*$": ""},
    "J Drugs Dermatol": {"\s*J Drugs Dermatol\. [\s0-9,;:\-\.\(\)]*$": ""},
    "Infect Control Hosp": {"\s*Infect Control Hosp Epidemiol [\s0-9,;:\-\.\(\)]*$": ""},
    "Magn. Reson.": {"\.\s*[0-9]\s*J\. Magn\. Reson\. Imaging [\s0-9,;:\-\.\(\)]*$": ""},
    "MAGN. RESON.": {"\.\s*[0-9]\s*J\. MAGN\. RESON\. IMAGING [\s0-9,;:\-\.\(\)]*$": ""},
    "Magnetic Resonance": {
        "\s*Magnetic Resonance in Medicine published by Wiley Periodicals\.*": "",
    },
    "(Pediatr Dent": {"\s*\(Pediatr Dent 20.*": ""},
    "Environ Toxicol Chem": {"\s*Environ Toxicol Chem [\s0-9,;:\-\.\(\)]*$": ""},
    "Environ Health Perspect": {"\s*Environ Health Perspect [\s0-9,;:\-\.\(\)]*$": ""},
    "Antioxid. Redox Signal.": {"\s*Antioxid\. Redox Signal\. [\s0-9,;:\-\.\(\)]*$": ""},
    "J Orthop Sports Phys Ther": {"\s*J Orthop Sports Phys Ther\.? [\sA0-9,;:\-\.\(\)]*$": ""},
    "J Strength Cond Res": {".*J Strength Cond Res.{0,20}20[012][0-9]-([A-Z])": "\\1"},
    "Turk J Pediatr": {".*Turk J Pediatr [\s0-9,;:\-\.\(\)]*([A-Z])": "\\1"},
    "Laryngoscope": {
        "\s*[1-9][a-zA-Z]?\. Laryngoscope[^\.]*\.\s*$": "",
        "[\.][^\.]*Laryngoscope[^\.]*\.\s*$": ".",
        "\s*N/A\.$": "",
    },
    "Indian J Crit Care Med": {
        # Author list. Title. Indian J Crit Care Med
        "\s*[^\.]*\.[^\.]*[\.?!] Indian J Crit Care Med [\s0-9,;:\-\.\(\)]*$": "",
    },
    "Int J Clin Pediatr Dent": {
        # Author list. Title. Int J Clin Pediatr Dent
        "\s*[^\.]*\.[^\.]*[\.?!] Int J Clin Pediatr Dent [\s0-9,;:\-\.\(\)]*$": "",
    },
    "J Clin Sleep Med": {
        # Author list. Title. Int J Clin Pediatr Dent
        "\s*[^\.]*\.[^\.]*[\.?!] J Clin Sleep Med\. [\s0-9,;:\-\.\(\)]*$": "",
    },
    "Hepatology": {
        "\(Hepatology [\s0-9,;:\-\.\(\)]*\)\.?\s*$": "",
        "\(Hepatology Communications [\s0-9,;:\-\.\(\)]*\)\.?\s*$": "",
    },
    "AASLD": {"[^\.]*AASLD\.\s*$": ""},
    "Database Record": {
        "\s*\(PsycINFO Database Record.*": "",
        "\s*\(PsycInfo Database Record.*": "",
    },
    "advance online publication": {
        "[^\.]* advance online publication,?\s*[0-9][0-9] [A-Za-z]* [0-9]{4}[\.;,]?\s*$": ""
    },
    "This article is protected": {"\sThis article is protected by copyright.*": ""},
    "This article is part": {
        "\s*This article is part of the themed issue.*": "",
        "\s*This article is part of a themed issue.*": "",
        "\s*This article is part of a themed section.*": "",
        "\s*This article is part of a Special Issue entitled.*": "",
    },
    "Elsevier Ltd": {"\s*20[012][0-9] Elsevier Ltd.*": ""},
    "How to cite this article:": {"\s*How to cite this article:.*": ""},
    "Cite this article:": {"\s*Cite this article:.*": ""},
    "Citation": {"\s+Citation: .*": ""},
    "ClinicalTrials.gov:": {"\s*\(?ClinicalTrials.gov: .{0,100}$": ""},
    ".].": {"\[[^\[]*[0-9]\.\]\.?$": ""},
    "https://youtu.be": {
        "\.\s*[^.]*: https://youtu.be.{0,100}$": ".",
        "\. https://youtu.be.{0,50}.$": ".",
    },
    "The virtual slide(s) for this article": {"\s*The virtual slide\(s\) for this article.*": ""},
    "IMPACT STATEMENT": {"\.\s*IMPACT STATEMENT[A-Z].*": "."},
    "Impact statement": {"\.\s*Impact Statement[A-Z].*": "."},
    ("RESULTS", "CONCLUSIONS"): {
        "\.\s*PURPOSE[.: ][.:]?\s*([A-Z])": ". \\1",
        "\.\s*BACKGROUND[.: ][.:]?\s*([A-Z])": ". \\1",
        "\.\s*INTRODUCTION[.: ][.:]?\s*([A-Z])": ". \\1",
        "\.\s*OBJECTIVE[.: ][.:]?\s*([A-Z])": ". \\1",
        "\.\s*MATERIALS AND METHODS[.: ][.:]?\s*([A-Z])": ". \\1",
        "\.\s*MATERIALS & METHODS[.: ][.:]?\s*([A-Z])": ". \\1",
        "\.\s*METHODS?[.: ][.:]?\s*([A-Z])": ". \\1",
        "\.\s*METHODOLOGY[.: ][.:]?\s*([A-Z])": ". \\1",        
        "\.\s*DESIGN[.: ][.:]?\s*([A-Z])": ". \\1",
        "\.\s*STUDY DESIGN[.: ][.:]?\s*([A-Z])": ". \\1",
        "\.\s*KEY RESULTS[.: ][.:]?\s*([A-Z])": ". \\1",
        "\.\s*RESULTS[.: ][.:]?\s*([A-Z])": ". \\1",
        "\.\s*CONCLUSIONS[.: ][.:]?\s*([A-Z])": ". \\1",
        "\.\s*CONCLUSIONS AND INFERENCES[.: ][.:]?\s*([A-Z])": ". \\1",
        "\.\s*CONCLUSIONS & INFERENCES[.: ][.:]?\s*([A-Z])": ". \\1",
    },
    ("Results", "Conclusions"): {
        "\.\s*Purpose[.: ][.:]?\s*([A-Z])": ". \\1",
        "\.\s*Background[.: ][.:]?\s*([A-Z])": ". \\1",
        "\.\s*Introduction[.: ][.:]?\s*([A-Z])": ". \\1",
        "\.\s*Objective[.: ][.:]?\s*([A-Z])": ". \\1",
        "\.\s*Materials and methods[.: ][.:]?\s*([A-Z])": ". \\1",
        "\.\s*Materials and Methods[.: ][.:]?\s*([A-Z])": ". \\1",
        "\.\s*Materials & Methods[.: ][.:]?\s*([A-Z])": ". \\1",
        "\.\s*Methods?[.: ][.:]?\s*([A-Z])": ". \\1",
        "\.\s*Methodology[.: ][.:]?\s*([A-Z])": ". \\1",
        "\.\s*Design[.: ][.:]?\s*([A-Z])": ". \\1",
        "\.\s*Study Design[.: ][.:]?\s*([A-Z])": ". \\1",
        "\.\s*Key Results[.: ][.:]?\s*([A-Z])": ". \\1",
        "\.\s*Results[.: ][.:]?\s*([A-Z])": ". \\1",
        "\.\s*Conclusions[.: ][.:]?\s*([A-Z])": ". \\1",
        "\.\s*Conclusions and inferences[.: ][.:]?\s*([A-Z])": ". \\1",
        "\.\s*Conclusions and Inferences[.: ][.:]?\s*([A-Z])": ". \\1",
        "\.\s*Conclusions & inferences[.: ][.:]?\s*([A-Z])": ". \\1",
        "\.\s*Conclusions & Inferences[.: ][.:]?\s*([A-Z])": ". \\1",
    },
    "Expert commentary": {"\.\s*Expert commentary:\s*": ". ", "\.\s*Areas covered:\s*": ". "},
    "Details of funding": {"\s*Details of funding are provided.*": ""},
    "This journal requires": {"\s*This journal requires.*": ""},
    "Proprietary or commercial disclosure": {
        "\s*Proprietary or commercial disclosure.*": ""
    },
    "See acknowledgments": {".\s*See acknowledgments.\s*$": "."},
    "This article is one of ten": {
        "\s*This article is one of ten reviews.*": ""
    },
    "In an effort to expedite the publication of articles": {
        "^In an effort to expedite the publication of articles.*": ""
    },
    "For complete coverage": {
        "\s*For complete coverage of all related areas of Endocrinology.*": ""
    },
    "Abbreviations": {"\.\s*Abbreviations:.*": ".", "\.\s*Abbreviations [Uu]sed:.*": "."},
    "ABBREVIATIONS": {"\.\s*ABBREVIATIONS:.*": ".", "\.\s*ABBREVIATIONS USED:.*": "."},
    "Registration number": {"\s*Registration number of the clinical trial:.*": ""},
    " de ": {
        "\.\s*:?\s*[Aa]nalisar .*": ".",
        "\.\s*:?\s*[Dd]escrever .*": ".",
        "\.\s*:?\s*[Ii]mplementar .*": ".",
        "\.\s*:?\s*[Cc]ompreender .*": ".",
        "\.\s*:?\s*[Aa]valiar .*": ".",
        "\.\s*:?\s*[Ee]stimar .*": ".",
        "\.\s*:?\s*[Dd]eterminar .*": ".",
        "\.\s*:?\s*[Rr]ealizar .*": ".",
        "\.\s*:?\s*[Cc]aracterizar .*": ".",
        "\.\s*:?\s*[Ii]dentificar .*": ".",
        "\.\s*:?\s*[Dd]iscutir .*": ".",
        "\.\s*:?\s*[Cc]onhecer .*": ".",
        "\.\s*:?\s*[Cc]onocer .*": ".",
        "\.\s*:?\s*Resumo .*": ".",
    },
}

all_affected_abstracts = np.zeros(len(df), dtype=bool)
  
for search_string in to_replace:
    if type(search_string) == str:
        print(f'Searching for: {search_string}', end='', flush=True)
        ind = df.AbstractText.str.contains(search_string, regex=False)
    else:
        print('Searching for: ' + ' + '.join(search_string), end='', flush=True)
        ind = np.ones(len(df), dtype=bool)
        for search_str in search_string:
            ind &= df.AbstractText.str.contains(search_str, regex=False)        
    
    print(f' --> found {np.sum(ind)} abstracts.', flush=True)
    
    for replace_string in to_replace[search_string]:
        s = df[ind].AbstractText.str.extract('(' + replace_string + ')').values[:, 0]
        
        ind2 = [type(ss) == str for ss in s.ravel()]
        all_affected_abstracts[np.where(ind)[0][ind2]] = True
        
        s = [ss[:75] for ss in s.ravel() if type(ss) == str]
        if len(s) > 0:
            print(f'   Found {len(s)} abstract(s) with string(s) to replace:')
            print('      ' + '\n      '.join(s[:5]) + '\n', flush=True)
        else:
            print('   Found nothing to replace.\n')
        df.loc[ind, 'AbstractText'] = df[ind].AbstractText.str.replace(
            replace_string, 
            to_replace[search_string][replace_string], 
            regex=True
        )
        
print(f'In total {np.sum(all_affected_abstracts)} were edited.')

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(binary=True)
X = vectorizer.fit_transform(df.AbstractText.values) # ~30 min # 14_182_520, 4_119_741

print(f'Count matrix computed: {X.shape}')

words = vectorizer.get_feature_names_out()
years = np.arange(2010, 2025)
counts = np.zeros((words.size, years.size))
totals = np.zeros(years.size)

for i, year in enumerate(years):
    ind = df.Year == year
    counts[:, i] = np.array(np.sum(X[ind, :], axis=0)).ravel()
    totals[i] = np.sum(ind)
    
import pickle
pickle.dump([words, years, counts, totals], open('yearly-counts.pkl', 'wb'))

chatgpt_words = np.array([
       'accentuates', 'achieving', 'acknowledging', 'across',
       'additionally', 'address', 'addresses', 'addressing', 'adept',
       'adhered', 'advancement', 'advancements', 'advancing',
       'advocating', 'affirming', 'afflicted', 'aiding', 'akin', 'align',
       'aligning', 'aligns', 'alongside', 'amplifies', 'approach',
       'assess', 'augmenting', 'avenue', 'avenues', 'bolster',
       'bolstered', 'bolstering', 'burgeoning', 'capabilities',
       'capitalizing', 'categorizing', 'challenges', 'commendable',
       'compelling', 'comprehend', 'comprehended', 'comprehending',
       'comprehensive', 'consequently', 'consolidates', 'contributing',
       'conversely', 'crafting', 'crucial', 'culminating', 'delineates',
       'delve', 'delved', 'delves', 'delving', 'demonstrated',
       'demonstrates', 'demonstrating', 'dependable', 'detrimentally',
       'diminishes', 'discern', 'discerned', 'discernible', 'discerning',
       'displaying', 'distinct', 'distinctions', 'distinctive', 'diverse',
       'elevates', 'elevating', 'elucidate', 'elucidates', 'elucidating',
       'emerged', 'emerges', 'emphasises', 'emphasising', 'emphasize',
       'emphasizes', 'emphasizing', 'employed', 'employing', 'employs',
       'empowers', 'enabling', 'encapsulates', 'encompass', 'encompassed',
       'encompasses', 'encompassing', 'endeavors', 'endeavours',
       'enduring', 'enhance', 'enhancements', 'enhances', 'enhancing',
       'ensuring', 'escalating', 'exacerbating', 'exceeding', 'excels',
       'exceptional', 'exceptionally', 'exhibit', 'exhibited',
       'exhibiting', 'exhibits', 'expedite', 'expediting', 'exploration',
       'explores', 'facilitated', 'facilitating', 'featuring', 'findings',
       'focusing', 'formidable', 'forthcoming', 'fostering', 'fosters',
       'foundational', 'garnered', 'gauged', 'grappling',
       'groundbreaking', 'groundwork', 'harness', 'harnesses',
       'harnessing', 'heightened', 'highlighting', 'highlights', 'hinges',
       'hinting', 'hold', 'holds', 'illuminates', 'illuminating',
       'impact', 'impacting', 'impede', 'impeding', 'imperative',
       'impressive', 'inadequately', 'including', 'incorporates',
       'incorporating', 'inherent', 'innovative', 'inquiries', 'insights',
       'integrates', 'interconnectedness', 'interplay', 'into',
       'intricacies', 'intricate', 'intricately', 'intriguing',
       'introduces', 'invaluable', 'involves', 'juxtaposed', 'leverages',
       'leveraging', 'merges', 'meticulous', 'meticulously',
       'multifaceted', 'necessitate', 'necessitates', 'necessitating',
       'necessity', 'notable', 'notably', 'noteworthy', 'nuanced',
       'nuances', 'observed', 'offer', 'offering', 'offers', 'optimizing',
       'orchestrating', 'outcomes', 'overlooking', 'particularly',
       'paving', 'pinpoint', 'pinpointed', 'pinpointing', 'pioneering',
       'pioneers', 'pivotal', 'poised', 'posed', 'poses', 'posing',
       'potential', 'precise', 'pressing', 'primarily', 'promise',
       'prompting', 'propelling', 'realm', 'realms', 'refine', 'refining',
       'remains', 'remarkable', 'renowned', 'revealed', 'revealing',
       'revolutionize', 'revolutionizing', 'revolves', 'scrutinize',
       'scrutinized', 'scrutinizing', 'seamless', 'seamlessly', 'serves',
       'shedding', 'sheds', 'showcased', 'showcases', 'showcasing',
       'signifying', 'spanned', 'spanning', 'spurred', 'stands',
       'strategically', 'streamline', 'streamlines', 'streamlining',
       'subsequently', 'substantial', 'substantiated', 'substantiates',
       'substantiating', 'surmount', 'surpassed', 'surpasses',
       'surpassing', 'swift', 'thereby', 'these', 'thorough', 'through',
       'transformative', 'uncharted', 'uncovering', 'underexplored',
       'underscore', 'underscored', 'underscores', 'underscoring',
       'understanding', 'unraveling', 'unveil', 'unveiled', 'unveiling',
       'unveils', 'uphold', 'upholding', 'urging', 'utilizes',
       'utilizing', 'valuable', 'various', 'varying', 'versatility',
       'warranting', 'while', 'within'
])

chatgpt_words_f = np.array([
       2.01184755e-04, 1.80924027e-02, 9.97795096e-04, 8.35262201e-02,
       6.50964772e-02, 3.84506742e-02, 5.29176870e-03, 1.67918144e-02,
       2.54021155e-04, 1.47129053e-03, 6.45823384e-03, 9.92714673e-03,
       5.87296910e-03, 8.88057957e-04, 1.04859933e-03, 6.36068972e-04,
       1.98746152e-03, 8.18964203e-04, 2.71497810e-03, 1.62167105e-03,
       1.30262048e-03, 6.68990113e-03, 5.79168233e-04, 1.06751882e-01,
       8.69321357e-02, 1.46519402e-03, 4.10701403e-03, 5.02961887e-03,
       7.47838280e-04, 2.01184755e-04, 4.65366756e-04, 6.44197649e-04,
       1.03254519e-02, 2.70278509e-04, 7.64095634e-04, 5.01823872e-02,
       4.61302417e-04, 2.45079610e-03, 1.48958005e-03, 1.68670047e-04,
       1.23962324e-03, 5.44011705e-02, 1.55481268e-02, 2.78407186e-04,
       1.64646352e-02, 8.94764116e-03, 3.33275755e-04, 5.92926019e-02,
       7.88481665e-04, 6.27940295e-04, 1.81675930e-03, 6.64519341e-04,
       3.20676306e-03, 5.03977971e-04, 1.03433350e-01, 2.32845951e-02,
       1.37476249e-02, 4.34884217e-04, 1.74766555e-04, 8.88057957e-04,
       1.37374641e-03, 4.73495433e-04, 1.23962324e-03, 4.43012894e-04,
       3.32462888e-03, 3.49187640e-02, 8.83993619e-04, 5.08651961e-03,
       3.69509333e-02, 5.52750033e-04, 1.26807361e-03, 1.46214577e-02,
       1.79237327e-03, 3.98305171e-03, 2.26891696e-02, 3.65587246e-03,
       6.29972464e-04, 6.60455003e-04, 6.45213733e-03, 6.71428717e-03,
       8.74442424e-03, 4.15924078e-02, 1.30973307e-02, 3.65587246e-03,
       3.51565278e-04, 1.30140118e-02, 2.80439355e-04, 2.41421706e-03,
       2.85316561e-03, 2.66823821e-03, 7.19997561e-03, 1.47332270e-03,
       2.13377770e-04, 1.14004694e-03, 4.35331294e-02, 1.53022344e-03,
       1.20162167e-02, 2.65482589e-02, 7.98439294e-03, 1.63386407e-03,
       1.06688885e-03, 4.05417763e-03, 2.82471524e-04, 5.69413821e-03,
       1.64199275e-03, 2.56378471e-02, 6.25582979e-02, 8.78100328e-03,
       1.87752116e-02, 1.03437414e-03, 4.02369509e-04, 1.27843767e-02,
       1.06668563e-02, 8.89277259e-03, 1.11606735e-02, 3.17831269e-03,
       1.61965921e-01, 1.62573539e-02, 1.43877582e-03, 4.83656279e-04,
       2.90600201e-03, 6.01522095e-04, 1.68263613e-03, 3.01167481e-03,
       2.94664540e-04, 2.29635124e-04, 7.05162726e-04, 1.19085117e-03,
       1.23555890e-03, 3.17018401e-04, 1.63996058e-03, 5.65146265e-03,
       1.65235681e-02, 2.75887296e-02, 4.43012894e-04, 2.66214170e-04,
       7.14510704e-03, 9.84789213e-03, 5.32428341e-04, 4.91784956e-04,
       1.07641972e-01, 5.35679811e-03, 1.91633559e-03, 1.22743022e-03,
       6.08431470e-03, 2.80845789e-03, 9.89666419e-04, 1.85807330e-01,
       3.10718677e-03, 1.21462755e-02, 6.84231383e-03, 1.58996921e-02,
       4.06433848e-04, 4.96255728e-02, 3.41810866e-03, 2.68246340e-04,
       8.83993619e-03, 1.93119075e-01, 1.17052948e-03, 8.95170550e-03,
       8.37253726e-04, 2.48331081e-03, 6.78541309e-03, 1.34936037e-03,
       1.36175661e-02, 1.84927401e-04, 1.64199275e-03, 5.72462075e-03,
       1.86959570e-04, 1.50380524e-03, 1.43471148e-03, 4.96052511e-03,
       1.48958005e-03, 3.00761047e-03, 3.96679435e-03, 5.06416574e-03,
       1.18312893e-02, 3.06146296e-02, 4.50328703e-03, 2.43860309e-03,
       8.31157219e-04, 1.27327596e-01, 2.25530142e-02, 1.31359420e-02,
       2.21140657e-02, 7.94984606e-03, 7.47838280e-04, 1.26551307e-01,
       3.98305171e-04, 5.37264903e-02, 1.98136501e-03, 9.77473404e-04,
       4.97881464e-04, 3.84079986e-04, 1.38390725e-03, 2.66214170e-04,
       1.50319559e-02, 7.13291403e-04, 3.38762612e-03, 9.89259986e-03,
       2.99135312e-03, 2.14385726e-01, 1.77550626e-02, 2.70888160e-03,
       2.34532652e-02, 1.55786094e-02, 1.92852861e-03, 2.47924647e-04,
       2.38779886e-03, 3.29211417e-04, 2.57475843e-03, 1.59322068e-03,
       7.00793562e-02, 1.26929291e-02, 8.18964203e-04, 1.15041101e-01,
       8.68549133e-03, 1.04859933e-03, 5.46653525e-04, 1.74766555e-04,
       5.87296910e-04, 7.21420080e-04, 2.35731632e-04, 6.56390664e-04,
       7.64095634e-04, 8.21199590e-03, 3.01980349e-03, 2.84706910e-03,
       9.34797850e-04, 1.01202028e-03, 1.96307548e-03, 7.07194895e-04,
       6.80776695e-04, 4.89549570e-03, 4.32852048e-04, 3.12141195e-03,
       8.23028542e-04, 8.81961450e-04, 2.96696709e-04, 4.45045063e-04,
       2.55382708e-02, 2.62088867e-02, 1.00185943e-03, 2.37763801e-04,
       1.95088247e-04, 1.82895232e-04, 1.03234197e-03, 9.20572665e-04,
       2.44876393e-03, 8.41318065e-04, 2.60767957e-02, 3.36338234e-01,
       6.11276507e-03, 1.61230275e-01, 1.82895232e-03, 2.27602955e-04,
       1.40626111e-03, 1.53428778e-03, 7.64908502e-03, 1.47332270e-03,
       9.64061087e-03, 3.61726125e-03, 8.31177540e-02, 1.12175742e-03,
       1.85333835e-03, 3.17831269e-03, 7.47838280e-04, 1.12582176e-03,
       2.51988986e-04, 1.78830893e-04, 2.51988986e-04, 4.13546440e-03,
       2.08825711e-02, 3.36425618e-02, 1.06134103e-01, 1.82062042e-02,
       2.33089812e-03, 1.19085117e-03, 1.70730666e-01, 1.21434305e-01
])

cutoffs = [0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
cutoff_counts = np.zeros((len(cutoffs), years.size))
for i, cutoff in enumerate(cutoffs):
    ind_words = np.isin(words, chatgpt_words[chatgpt_words_f < cutoff])
    for j, year in enumerate(years):
        ind = df.Year == year
        cutoff_counts[i, j] = np.sum(np.sum(X[ind, :][:, ind_words], axis=1) > 0)
np.save('cutoff_counts.npy', cutoff_counts)

chatgptwords_common = [
    'exhibited', 'crucial', 'additionally', 'within', 'notably', 
    'insights', 'comprehensive', 'across', 'particularly', 'enhancing'
]

chatgptwords_rare = chatgpt_words[chatgpt_words_f < 0.01]

ind_words_common = np.isin(words, chatgptwords_common)
ind_words_rare = np.isin(words, chatgptwords_rare)

def count(subset_papers):
    group_counts = np.zeros((3, years.size), dtype=int)
    for i, year in enumerate(years):
        ind = subset_papers & (df.Year == year)
        for j, ind_words in enumerate([ind_words_common, ind_words_rare]):
            group_counts[j, i] = np.sum(np.sum(X[ind, :][:, ind_words], axis=1) > 0)
        group_counts[2, i] = np.sum(ind)
    return group_counts

def write_to_file(f, labeltype, label, group_counts):
    if "," in label:
        save_label = '"' + label + '"'
    else:
        save_label = label
    print(f'{labeltype},{save_label},' + ','.join(group_counts.ravel().astype(str)), file=f)
    print(f'{labeltype},{save_label},' + ','.join(group_counts.ravel().astype(str)))
    
with open('yearly-counts-subgroups.csv', 'w') as f:
    print(
        'grouptype,group,2010_common,2011_common,2012_common,2013_common,'
        '2014_common,2015_common,2016_common,2017_common,2018_common,'
        '2019_common,2020_common,2021_common,2022_common,2023_common,'
        '2024_common,2010_rare,2011_rare,2012_rare,2013_rare,2014_rare,'
        '2015_rare,2016_rare,2017_rare,2018_rare,2019_rare,2020_rare,'
        '2021_rare,2022_rare,2023_rare,2024_rare,2010_total,2011_total,'
        '2012_total,2013_total,2014_total,2015_total,2016_total,2017_total,'
        '2018_total,2019_total,2020_total,2021_total,2022_total,2023_total,'
        '2024_total',
        file=f
    )
    
    ind = np.ones(len(df), dtype=bool)
    write_to_file(f, 'all', 'all', count(ind))
    
    for label in set(np.unique(df.Label.values)) - set(['unlabeled']):
        ind = df.Label == label
        write_to_file(f, 'class', label, count(ind))
        
    countries, countries_counts = np.unique(df.Country.values, return_counts=True)
    countries = countries[np.argsort(countries_counts)][::-1]
    countries = [c for c in countries if c != 'unknown']
    for country in countries[:50]:
        ind = df.Country == country
        write_to_file(f, 'country', country, count(ind))

    for gender in ['male', 'female']:
        ind = df.Gender == gender
        write_to_file(f, 'gender', gender + ' first', count(ind))
        
    for gender in ['male', 'female']:
        ind = df.Gender_last == gender
        write_to_file(f, 'gender', gender + ' last', count(ind))
        
    journals, journals_counts = np.unique(df[df.Year == years[-1]].Journal.values, return_counts=True)
    journals = journals[np.argsort(journals_counts)][::-1]
    for journal in journals[:100]:
        ind = df.Journal == journal
        write_to_file(f, 'journal', journal, count(ind))
    
    ind = df.Journal.isin(['Nature', 'Science', 'Cell'])
    write_to_file(f, 'journals', 'Nature+Science+Cell', count(ind))
    
    # Established 2018 or earlier
    nature_journals = [
        'Nature aging', 'Nature astronomy', 'Nature biomedical engineering',
        'Nature biotechnology', 'Nature cardiovascular research', 'Nature catalysis',
        'Nature cell biology', 'Nature chemical biology', 'Nature chemistry',
        'Nature climate change', 'Nature communications', 'Nature computational science',
        'Nature digest', 'Nature ecology & evolution', 'Nature electronics',
        'Nature energy', 'Nature genetics', 'Nature geoscience', 'Nature human behaviour',
        'Nature immunology', 'Nature materials', 'Nature medicine', 'Nature methods',
        'Nature microbiology', 'Nature nanotechnology', 'Nature neuroscience',
        'Nature photonics', 'Nature physics', 'Nature plants', 
        'Nature structural & molecular biology', 'Nature sustainability'
    ]

    ind = df.Journal.isin(nature_journals)
    write_to_file(f, 'journals', 'Nature family', count(ind))
    
    ind = df.Journal.str.contains('Frontiers')
    write_to_file(f, 'journals', 'Frontiers', count(ind))
    
    ind = df.Journal.str.contains('Basel')
    write_to_file(f, 'journals', 'MDPI', count(ind))
    
    tadpole_journals = [
        'BioFactors (Oxford, England)',
        'Journal of cellular biochemistry',
        'Biomedicine & pharmacotherapy = Biomedecine & pharmacotherapie',
        'International immunopharmacology',
        'Experimental and molecular pathology',
        'Artificial cells, nanomedicine, and biotechnology',
        'Cellular physiology and biochemistry : international journal of experimental cellular physiology, biochemistry, and pharmacology',
        'International journal of immunopathology and pharmacology',
        'Brazilian journal of medical and biological research = Revista brasileira de pesquisas medicas e biologicas',
        'Die Pharmazie',
        'European review for medical and pharmacological sciences',
        'International journal of clinical and experimental pathology',
        'Neoplasma',
    ]
    
    ind = df.Journal.isin(tadpole_journals)
    write_to_file(f, 'journals', 'Tadpole journals', count(ind))
    
    ind = df.Title.str.contains('review') | df.Title.str.contains('Review')
    write_to_file(f, 'titles', 'Reviews', count(ind))
    
    for country in ['China', 'South Korea', 'Taiwan', 'Iran']:
        for label in ['computation', 'bioinformatics', 'material', 'healthcare', 'environment']:
            ind = (df.Country == country) & (df.Label == label)
            write_to_file(f, 'journal/class', country + ' & ' + label, count(ind))
            
    for country in ['China', 'South Korea', 'Taiwan', 'Iran']:
        for journal in ['Cureus', 'Sensors (Basel, Switzerland)']:
            ind = (df.Country == country) & (df.Journal == journal)
            write_to_file(f, 'journals/class', country + ' & ' + journal, count(ind))
            
    for country in ['China', 'South Korea', 'Taiwan', 'Iran']:
        for journal in ['Frontiers', 'Basel']:
            ind = (df.Country == country) & df.Journal.str.contains(journal)
            write_to_file(f, 'journals/class', country + ' & ' + journal, count(ind))
