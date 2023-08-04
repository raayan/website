# website
simple

## maybe not so
```bash
trunk build --release
ssh $HOST 'rm /var/www/website/web_test*'
scp dist/* $HOST:/var/www/website/
```
