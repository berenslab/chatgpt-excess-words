import pandas as pd
import xml.etree.ElementTree as et
import os

import numpy as np

def xml_import(xml_file):
    """Parses some elements of the metadata in PubMed XML files.
    Parses the following elements of each paper stored in the input `xml_file`, and stores them in a pandas 
    DataFrame. Elements parsed: PMID, Title, Abstract, Language, Journal, Date, First author name, Last authors name, ISSN.
    
    Parameters
    ----------
    xml_file : path/filename
        Filename of the XML file.
    
    Returns
    -------
    out : pandas dataframe of shape (n_papers, n_elements)
        Pandas dataframe containing all the parsed papers, and as columns the elements of the metadate that were parsed.
    dicc : dict
        Dictionary from which the pandas dataframe was created
    
    See Also
    --------
    xml_import
    
    Notes
    -----
    Parsed elements of each paper:
    - PMID (stored in <PMID>).
    - Title (stored in <ArticleTitle>).
    - Abstract (stored in <AbstractText>).
    - Language (stored in <Language>).
    - Journal (stored in <Title>).
    - Date (stored in <PubDate>).
    - First author first name (stored in <ForeName>, child of <AuthorList>, child of <Author>).
    - Last authors first name (stored in <ForeName>, child of <AuthorList>, child of <Author>).
    - ISSN (stored in <ISSN>)
    
    Details about information extracion: 
    - PMID: If there is no tag <PMID>, it will add 'no tag'. If there is more than one <PMID>,
    it will import only the first one. If <PMID> contains no text, it will add '' (empty string).
    
    - Title: If there is no tag <ArticleTitle>, it will add 'no tag'. If there is more than one
    <ArticleTitle>, it will import only the first one. If <ArticleTitle> contains no text, it will add '' (empty string).
    
    - Abstract: If there is no tag <Abstract> (parent of <AbstractText>), it will add '' (empty string).
    If there is more than one <AbstractText> inside <Abstract>, it will combine them into one list.
    If <AbstractText> contains no text, it will add '' (empty string).
    If there is more than one <Abstract> or other tags containing <AbstractText>, like 
    <OtherAbstract>, it will not get text from them. I am not sure but I think that with the fix it collected all the child tags from <Abstract>.
    
    - Language: If there is no tag <Language>, it will add 'no tag'. If there is more than one
    <Language>, it will import only the first one. If <Language> contains no text, it will add '' (empty string).
    
    - Journal: If there is no tag <Title>, it will add 'no tag'. If there is more than one <Title>,
    it will import only the first one. If <Title> contains no text, it will add '' (empty string).
    
    - Date: If there is no tag <PubDate>, it will add 'no tag'. It will combine all the <PubDate>'s childs' texts
    into one (due to the assymetry of the date storage, sometimes with <Day>, <Month> and <Year>, other times 
    with <MedlineDate>). If <PubDate> contains no further childs, it will print ' '.
    
    - First author first name: It parses the <ForeName> of the first <Author> listed in <Authorlist>. Note that sometimes the metadata is not perfect inside the tag there is the complete name, including surnames. If there is no tag <ForeName>, 'no tag' will be appended. Note for the future: Maybe this is misses some names directly listed in the tag <Author>, maybe an approach similar to what I do for abstracts would be better, where everything under the <Author> tag is parsed. In that case we would also have surnames, but that can be cleaned after. Update: actually not really because affiliation is also saved under <Author>. It is fine the way it is.
    
    - Last authors first name: It parses the <ForeName> of the last <Author> listed in <Authorlist>. Note that sometimes the metadata is not perfect inside the tag there is the complete name, including surnames. If there is no tag <ForeName>, 'no tag' will be appended. Note for the future: maybe same problem as in first authors.
    
    - ISSN (stored in <ISSN>): If there is no tag <ISSN>, it will add 'no tag'. If <ISSN> contains no text, it will add '' (empty string). 

    - Affiliation first author: 

    - Affiliation last author:

    """
    
    
    xtree = et.parse(xml_file, parser=et.XMLParser(encoding="UTF-8"))
    xroot = xtree.getroot()

    dicc={}

    #PMID 
    ros=[]
    for child1 in xroot:
        for child2 in child1:
            for element in child2.iter('MedlineCitation'):
                tag=element.find('PMID')
                if tag is None:
                    ros.append(['no tag'])
                else:
                    res=[]
                    if not tag.text :
                        res.append('')
                    else:
                        res.append(tag.text)
                    ros.append(res)


    ros=[' '.join(ele) for ele in ros]
    dicc['PMID']=ros

    
    #Title 
    ros=[]
    for child1 in xroot:
        for child2 in child1:
            for child3 in child2:
                for article in child3.iter('Article'):
                    tag=article.find('ArticleTitle')
                    if tag is None:
                        ros.append(['no tag'])
                    else:
                        res=[]
                        res.append("".join(tag.find(".").itertext()))
                        ros.append(res)

    ros=[' '.join(ele) for ele in ros]
    dicc['Title']=ros


    #Abstract 
    ros=[]
    for child1 in xroot:
        for child2 in child1:
            for child3 in child2:
                for article in child3.iter('Article'):
                    tag=article.find('Abstract')
                    if tag is None:
                        ros.append([''])
                    else:
                        for child4 in child3:
                            for elem in child4.iter('Abstract'):
                                res=[]
                                for AbstractText in elem.iter('AbstractText'):
                                    res.append("".join(AbstractText.find(".").itertext()).strip())
                                res=[' '.join(res)]
                                res=[elem.strip() for elem in res]
                                ros.append(res)


    #print(ros)
    ros=[' '.join(ele) for ele in ros] 
    dicc['AbstractText']=ros
    
    
    #Language 
    ros=[]
    for child1 in xroot:
        for child2 in child1:
            for child3 in child2:
                for article in child3.iter('Article'):
                    tag=article.find('Language')
                    if tag is None:
                        ros.append(['no tag'])
                    else:
                        res=[]
                        if not tag.text :
                            res.append('')
                        else:
                            res.append(tag.text)
                        ros.append(res)

    ros=[' '.join(ele) for ele in ros]
    dicc['Language']=ros

    
    #Journal 
    ros=[]
    for child1 in xroot:
        for child2 in child1:
            for child3 in child2:
                for child4 in child3:
                    for journal in child4.iter('Journal'):
                        tag=journal.find('Title')
                        if tag is None:
                            ros.append(['no tag'])
                        else:
                            res=[]
                            if not tag.text:
                                res.append('')
                            else:
                                res.append(tag.text)
                            ros.append(res)

    ros=[' '.join(ele) for ele in ros]
    dicc['Journal']=ros

    
    #Date
    ros=[]
    for child1 in xroot:
        for child2 in child1:
            for child3 in child2:
                for child4 in child3:
                    for child5 in child4:
                        for JI in child5.iter('JournalIssue'):
                            tag=JI.find('PubDate')
                            if tag is None:
                                ros.append(['no tag'])
                            else:
                                res=[]
                                for elem in tag:
                                    res.append(elem.text)
                                ros.append(res)
                                
    ros=[' '.join(ele) for ele in ros]
    dicc['Date']=ros
    
    
    #First name of the first author
    ros=[]
    for child1 in xroot:
        for child2 in child1:
            for child3 in child2:
                for child4 in child3.iter('Article'):
                    authorlist = child4.find('AuthorList')
                    if authorlist is None:
                        ros.append('')
                    else:
                        for elem in child4.iter('AuthorList'):
                            author = elem.find('Author')
                            tag=author.find('ForeName')

                            if tag is None:
                                ros.append("")
                            else:
                                res=[]
                                if not tag.text:
                                    res.append('')
                                else:
                                    res.append(tag.text)
                                ros.append(res)

    ros=[' '.join(ele) for ele in ros]
    dicc['NameFirstAuthor']=ros
    
    
    #First name of the last author
    ros=[]
    for child1 in xroot:
        for child2 in child1:
            for child3 in child2:
                for child4 in child3.iter('Article'):
                    authorlist = child4.find('AuthorList')
                    if authorlist is None:
                        ros.append('')
                    else:
                        for author in child4.iter('AuthorList'):
                            res=[]
                            for elem in author.iter('Author'):
                                tag=elem.find('ForeName')
                                if tag is None:
                                    res.append("") 
                                else:
                                    if not tag.text:
                                        res.append('')
                                    else:
                                        res.append(tag.text)
                            ros.append(res[-1])
                            
    dicc['NameLastAuthor']=ros
    
    
    #ISSN 
    ros=[]
    for child1 in xroot:
        for child2 in child1:
            for child3 in child2:
                for child4 in child3:
                    for journal in child4.iter('Journal'):
                        tag=journal.find('ISSN')
                        if tag is None:
                            ros.append(['no tag'])
                        else:
                            res=[]
                            if not tag.text:
                                res.append('')
                            else:
                                res.append(tag.text)
                            ros.append(res)

    ros=[' '.join(ele) for ele in ros]
    dicc['ISSN']=ros


    # Affiliation of the first author 
    ros = []
    for child1 in xroot:
        for child2 in child1:
            for child3 in child2:
                for child4 in child3.iter("Article"):
                    authorlist = child4.find("AuthorList")
                    if authorlist is None:
                        ros.append("")
                    else:
                        for elem in child4.iter("AuthorList"):
                            author = elem.find("Author")
                            affil = author.find("AffiliationInfo")
                            if affil is None:
                                ros.append("")  
                            else:
                                tag = affil.find("Affiliation")
                                if tag is None:
                                    ros.append("")  
                                else:
                                    res = []
                                    if not tag.text:
                                        res.append("")
                                    else:
                                        res.append(tag.text)
                                    ros.append(res)
    
    ros = [" ".join(ele) for ele in ros]
    dicc['AffiliationFirstAuthor']=ros

    
    # Affiliation of the last author
    ros = []
    for child1 in xroot:
        for child2 in child1:
            for child3 in child2:
                for child4 in child3.iter("Article"):
                    authorlist = child4.find("AuthorList")
                    if authorlist is None:
                        ros.append("")
                    else:
                        for author in child4.iter("AuthorList"):
                            res = []
                            for elem in author.iter("Author"):
                                affil = elem.find("AffiliationInfo")
                                if affil is None:
                                    res.append("")
                                else:
                                    tag = affil.find("Affiliation")
                                    if tag is None:
                                        res.append("")
                                    else:
                                        if not tag.text:
                                            res.append("")
                                        else:
                                            res.append(tag.text)
                            ros.append(res[-1])
                            
    dicc['AffiliationLastAuthor']=ros
    
    out=pd.DataFrame.from_dict(dicc)
    return out, dicc



def import_all_files(path, order_files=False):
    """Imports all xml files from a directory into a combined dataframe using the function xml_import.
    
    WARNING: I changed the name of the xml_import function that also includes the first names of first and last authors and ISSN, so now this function works calling the old function. A new import_all_files function needs to be created that calls the new xml_import_with_authors_ISSN.
    
    Parameters
    ----------
    path : srt 
        Path of the directory with the files you want to import.
    order_files : bool, default=False 
        If True, it will print the order in which files are being imported.
        
    Returns
    -------
    final_df : pandas dataframe
        Dataframe with all the XML files from the directory imported and merged together (concatenated in the order that they were in the directory (from up to down)).

    """
    # name_files has the names of both .xml files and .gz.md5 files
    name_files=os.listdir(path)
    
    # we select only the .xml files
    len_filenames_list=map(len, name_files)
    len_filenames=np.fromiter(len_filenames_list, dtype=np.int64,count=len(name_files))

    name_files_array=np.array(name_files)
    name_xml_files=name_files_array[len_filenames==17]
    name_xml_files.sort()
    
    # import
    frame_all_df=[]
    
    for i in range(0,len(name_xml_files)):
        path_file=path+name_xml_files[i]
        if order_files==True:
            print(name_xml_files[i])
        df,dic=xml_import(str(path_file))
        dic['filename'] = [name_xml_files[i]]*len(dic['Title'])
        df=pd.DataFrame.from_dict(dic)
        frame_all_df.append(df)

    final_df=pd.concat(frame_all_df,ignore_index=True)
    return final_df



def improved_coloring(journals, dict_words_colors):
    """ Creates coloring based on words appearing in a list of documents.
    It creates an array with colors, assigning a color to each paper depending on whether it contains a word in its journal title from the keys in ` dict_words_colors`. 
    
    IMPORTANT REMARK: if the journal name contains two words belonging to the word list, the color of the word
    located the latest in the list will be assigned to it (first, the first word's color is assigned and then 
    the second overwrites the first).
    
    Parameters
    ----------
    journals : dataframe of str
        Dataframe with the journal names of the papers, or any other corpus where to look for the words.
    dict_words_colors : dict
        Dictionary matching words to colors (legend). The keys are the words and the values are the colors.
    
    
    Returns
    -------
    labels_with_unlabeled : list of str fo len (n_journals)
        List or labels (words) for all instances including label 'unlabeled'.
    colors : array
        Colors for each paper.
            
    See Also
    --------
    automatic_coloring
    
    """
    
    
    words=dict_words_colors.keys()
    labels=np.empty(len(journals))
    
    for i, wrd in enumerate(words):
        
        word_may = wrd.capitalize()
        word_min = ' '+wrd
        
        indexes1 = journals.str.find(word_may) 
        indexes2 = journals.str.find(word_min)
        
        labels = np.where((indexes1!=-1) | (indexes2!=-1), wrd, labels)
    
    #create colors
    colors=np.vectorize(dict_words_colors.get)(labels)
    
    #add grey to the rest of papers
    colors=np.where(colors==None,'lightgrey', colors)
    colors=np.where(colors=='None','lightgrey', colors)
    
    #change 0 for 'unlabeled'
    labels_with_unlabeled=np.where(colors=='lightgrey','unlabeled', labels)
    
    
    return labels_with_unlabeled, colors



def mapping_countries(affiliations, dict_countries):
    """Maps countries to affiliation strings.
    The affiliation strings include the country name, so it searches for all possible country names in the strings.
    This produces a list of countries, correcting and assigning only one out of all possible different country names (e.g., "US" and "USA").

    Parameters
    ----------
    affiliations : dataframe of str
        Dataframe with the affiliation names of the papers.
    dict_countries : dict
        Dictionary matching country to number (legend).


    Returns
    -------
    labels : list of str fo len (affiliations)
        List of countries for all papers including 'unknown'.
    numbers : array
        Numbers for each paper. Each number corresponds to a country.

    """

    # create inverse dict
    inv_dict = {v: k for k, v in dict_countries.items()}
    inv_dict[0] = "unknown"

    # add special cases
    dict_countries["United States of America"] = dict_countries[
        "United States"
    ]
    dict_countries["USA"] = dict_countries["United States"]
    dict_countries["America"] = dict_countries["United States"]
    dict_countries["UK"] = dict_countries["United Kingdom"]
    dict_countries["Republic of Korea"] = dict_countries["South Korea"]
    dict_countries["Korea"] = dict_countries["South Korea"]

    countries = dict_countries.keys()
    numbers = np.zeros(len(affiliations))

    for country in countries:
        country_str = " " + country
        country_upper = country.upper()

        indexes1 = affiliations.str.find(country_str)
        indexes2 = affiliations.str.find(country_upper)

        numbers = np.where(
            (indexes1 != -1) | (indexes2 != -1),
            dict_countries[country],
            numbers,
        )

    # create labels
    labels = np.vectorize(inv_dict.get)(numbers)

    # add unknown to the rest of papers
    labels = np.where(labels == None, "unknown", labels)
    labels = np.where(labels == "None", "unknown", labels)

    return labels, numbers


def mapping_states(affiliations, dict_countries):
    """Maps USA states to affiliation strings.

    Parameters
    ----------
    affiliations : dataframe of str
        Dataframe with the affiliation names of the papers.
    dict_countries : dict
        Dictionary matching country to number (legend).


    Returns
    -------
    labels : list of str fo len (affiliations)
        List of states for all papers including 'unknown'.
    numbers : array
        Numbers for each paper. Each number corresponds to a country.

    See Also
    --------
    mapping_countries

    """
    # create inverse dict
    inv_dict = {v: k for k, v in dict_countries.items()}
    inv_dict[0] = "unknown"

    countries = dict_countries.keys()
    numbers = np.zeros(len(affiliations))

    for country in countries:
        country_str = " " + country
        country_upper = country.upper()

        indexes1 = affiliations.str.find(country_str)
        indexes2 = affiliations.str.find(country_upper)

        numbers = np.where(
            (indexes1 != -1) | (indexes2 != -1),
            dict_countries[country],
            numbers,
        )

    # create labels
    labels = np.vectorize(inv_dict.get)(numbers)

    # add unknown to the rest of papers
    labels = np.where(labels == None, "unknown", labels)
    labels = np.where(labels == "None", "unknown", labels)

    return labels, numbers