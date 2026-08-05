"""Convenience launcher intended for the Google Colab runtime."""

from app import build_app


if __name__ == "__main__":
    build_app().launch(share=True)

