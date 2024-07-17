import sys

from chatbot import Chatbot

from document_saver import PdfEmbedder

import tkinter as tk
from tkinter import ttk

from PIL import ImageTk, Image


class App(object):

    def __init__(self):

        self.pdf_saver = PdfEmbedder()
        self.chatbot = Chatbot()

        self.main_window = tk.Tk()
        self.main_window.title("Product manual chatbot")

        self.main_frame = ttk.Frame(self.main_window, padding="3 3 12 12")
        self.main_frame.grid(column=0, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))

        self.response_frame = ttk.Frame(self.main_frame, padding="3 3 12 12")
        self.response_frame.grid(column=1, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))

        self.image_frame = ttk.Frame(self.response_frame, padding="3 3 12 12")
        self.image_frame.grid(column=1, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))

        self.answer = tk.StringVar()

        self.answer_label = ttk.Label(self.response_frame, textvariable=self.answer)
        self.image_label = ttk.Label(self.response_frame)

        self.answer_label.grid(column=1, row=1, sticky=(tk.N, tk.S))

        self.image_label.grid(column=1, row=0, sticky=(tk.N, tk.S))
        self.image_label.grid_propagate(False)

        self.main_window.columnconfigure(0, weight=1)
        self.main_window.rowconfigure(0, weight=1)

        self.question_label = ttk.Label(
            self.main_frame, text="Ask a question about a product!"
        )
        self.question_label.grid(column=0, row=1, sticky=(tk.W, tk.E))

        self.question = tk.StringVar()

        self.question_entry = ttk.Entry(
            self.main_frame, width=7, textvariable=self.question
        )

        self.question_entry.grid(column=1, row=1, sticky=(tk.W, tk.E))

        self.ask_button = ttk.Button(
            self.main_frame, text="ASK", command=self.ask_question
        )
        self.ask_button.grid(column=1, row=2, sticky=(tk.W, tk.E))

        for child in self.main_frame.winfo_children():
            child.grid_configure(padx=5, pady=5)

        self.main_window.mainloop()

        return

    def ask_question(self):

        question = self.question.get()

        answer, image_path = self.chatbot.ask(question)
        self.question.set("")

        image = ImageTk.PhotoImage(Image.open(image_path))

        self.image_label.configure(image=image)
        self.image_label.image = image

        self.answer.set(answer)

        self.main_window.update()


if __name__ == "__main__":

    app = App()
