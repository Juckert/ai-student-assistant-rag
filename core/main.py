def main():
    ensure_runtime_state()
    user = get_current_user()

    if user is None:
        render_login_screen()
        return

    render_sidebar(user)

    if user["role"] == "admin":
        render_admin_panel(user)
    else:
        render_student_chat(user)


if __name__ == "__main__":
    main()