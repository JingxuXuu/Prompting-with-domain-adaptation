data_dir = os.path.join('data', args.root_data_dir, domain)
    sentences = []
    labels = []
    train_path = os.path.join("data", "amazon_review", domain, "negative.parsed")
    neg_tree = ET.parse(train_path)
    neg_root = neg_tree.getroot()
    for rev in neg_root.iter('review'):
        labels.append(0)
        sentences.append(' '.join(rev.text))
    train_path = os.path.join("data", "amazon_review", domain, "positive.parsed")
    pos_tree = ET.parse(train_path)
    pos_root = pos_tree.getroot()
    for rev in pos_root.iter('review'):
        labels.append(1)
        sentences.append(' '.join(rev.text))
    return sentences, labels