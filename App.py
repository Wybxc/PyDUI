def App(__ctx__, __id__):
    with __ctx__.group(2):
        with Window(__ctx__, 3) as w:
            w.title = 'PyDUI'
            w.size = (250, 150)
        with VBox(__ctx__, 4):
            count = __ctx__.use_state(5, 0)
            if count() == 0:
                with __ctx__.group(8):
                    with Label(__ctx__, 6, 'Hello PyDUI!', alignment='center'):
                        pass
            else:
                with __ctx__.group(9):
                    with Counter(__ctx__, 7, count(), alignment='center'):
                        pass
            with Button(__ctx__, 10, text='Click Me') as btn:

                def __event_handler_11__():
                    count(count() + 1)
                btn.on_click = __event_handler_11__