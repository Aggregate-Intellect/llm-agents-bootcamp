from pyboxen import boxen

def print_debug(title: str, content: str, color: str = "blue"):
    print(boxen(content, title=f">>> {title}", color=color, padding=(1, 2)))