import subprocess
import os

class RnnTagger(object):
    """Create an instance of rnntagger
    Takes the path of rnntagger (No need for the language as we only will be using it in english)"""

    def __init__(self):
        self.script = 'cmd/rnn-tagger-english.sh'
        self.fileIN = "in.txt"
        self.fileOUT = "out.txt"
        self.pos_tag = list()
        self.lemmas = list()
        self.tokens = list()


    def tag(self, text):
        """Gets a list of tokens of the sentence"""
        # put sentence in standard file
        os.chdir("./RNNTagger/")

        f = open(self.fileIN, "w")
        f.write(text)
        f.close()

        # Run RNNTagger
        cmd = [self.script, self.fileIN]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        for line in p.stdout:
            line = line.replace("\n", "")
            if line != "":
                self.tokens.append(line.split("\t")[0].lower())
                self.lemmas.append(line.split("\t")[2].lower())
                self.pos_tag.append(line.split("\t")[1].lower())
        p.wait()
        #print p.returncode
        # Read out file and put every line in list
        os.chdir("..")
        return 0

    def lemmas(self):
        """"Gets the lemmas"""
        return self.lemmas

    def tokens(self):
        """"Gets the tokens"""
        return self.tokens

    def pos_tag(self):
        """"Gets the pos_tag"""
        return self.pos_tag
