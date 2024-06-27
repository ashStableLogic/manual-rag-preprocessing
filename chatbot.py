from database import Database


class Chatbot(object):

    def __init__(self) -> None:

        self.db = Database()

    def ask(self, question: str) -> str:

        most_relavent_embedding = self.db.get_most_relavent_embeddings(question)

        return most_relavent_embedding


def main():

    chatbot = Chatbot()

    while True:
        question = input("Ask a question about CDJ-3000 or R4731 1:\t")

        answer = chatbot.ask(question)

        print(answer)


if __name__ == "__main__":

    main()
