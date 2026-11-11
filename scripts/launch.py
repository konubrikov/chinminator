from one_click import main
import sys
if __name__ == '__main__':
    try: raise SystemExit(main())
    except Exception as exc:
        print('START FAILED:',exc,file=sys.stderr)
        raise SystemExit(1)
