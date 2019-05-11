class Context(object):
    def __init__(self, subject, text):
        self.subject = subject
        self.text = text


def convert_to_document(text, subject):
    document = {
        'subject': subject,
        'text': text
    }
    return document


def convert_to_context(document):
    return Context(subject=document['subject'],
                   text=document['text'])
