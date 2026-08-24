# Butler

Внешний агент-дворецкий: голосовухи владельца в Telegram → голова (claude/codex)
с полными правами. Мост — тупой транспорт, весь интеллект в персоне и скиллах.
Спека — `SPEC.md`, решения — `DESIGN.md`.

## Установка

```bash
cd /home/dev/butler
python3.12 -m venv .venv && .venv/bin/pip install -e ".[dev]"
cp .env.example .env    # заполнить: токен бота, admin id, GROQ_API_KEY
ln -sfn /home/dev/secretary/skills/roles/secretary/open-sprint   skills/open-sprint
ln -sfn /home/dev/secretary/skills/roles/secretary/spec-card     skills/spec-card
ln -sfn /home/dev/secretary/skills/roles/secretary/knowledge-doc skills/knowledge-doc
ln -sfn packaging/butler.service ~/.config/systemd/user/butler.service
systemctl --user daemon-reload && systemctl --user enable --now butler
```

Проверка: `python -m butler_bridge` без `.env` падает с внятной ошибкой,
с заполненным — начинает long polling. `.env` читается из корня репо самим
конфигом (переменные окружения приоритетнее; путь переопределяется `BUTLER_ENV_FILE`),
так что запуск руками и под systemd ведут себя одинаково. Логи: `journalctl --user -u butler -f`.
`secretary reconcile` в установке не участвует.

`systemctl --user restart butler` можно делать на живом мосту: по SIGTERM он перестаёт
принимать сообщения, договаривает идущий тёрн и только потом выходит (§4.9 спеки).
Ждёт он не дольше `BUTLER_SHUTDOWN_GRACE_S`; `TimeoutStopSec` в юните заведомо больше,
поэтому меняешь порог в `.env` — правь и юнит, иначе systemd прибьёт мост посреди ответа.

## Тесты

```bash
.venv/bin/pytest -q && .venv/bin/ruff check .
```
