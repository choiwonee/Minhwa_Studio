import base64
import os
import sys

try:
    from configobj import ConfigObj
    from Crypto.Cipher import AES
    from Crypto.Util.Padding import pad
    from Crypto.Util.Padding import unpad 
except ImportError:
    print("Required libraries not found. Please install 'configobj' and 'pycryptodome'.")
    sys.exit(1)

# -----------------------------------------------------------------------------
# AES Encryption Logic
# -----------------------------------------------------------------------------
def prepare_KEY(key, length=32):
    if len(key) < length:
        return key.ljust(length, b'\0')
    return key[:length]

aes_KEY = prepare_KEY(b'1qazxsw23edcvfr45tgbnhy67ujm,ki8') 
aes_IV  = prepare_KEY(b'0p;/.lo98ik,mju7', 16)

def encrypt_AES(plain_text, key, iv):
    if not plain_text: return ""
    cipher = AES.new(key, AES.MODE_CBC, iv)
    padded_data = pad(plain_text.encode('utf-8'), AES.block_size)
    encrypted = cipher.encrypt(padded_data)
    return base64.b64encode(encrypted).decode('utf-8')

def decrypt_AES(encrypted_text, key, iv):
    if not encrypted_text: return ""
    try:
        cipher = AES.new(key, AES.MODE_CBC, iv)
        decoded = base64.b64decode(encrypted_text)
        decrypted_padded = cipher.decrypt(decoded)
        try:
            decrypted = unpad(decrypted_padded, AES.block_size).decode('utf-8')
        except ValueError:
            decrypted = decrypted_padded.decode('utf-8').strip('\x00') 
        return decrypted.strip()
    except Exception as e:
        raise e

# -----------------------------------------------------------------------------
# Config & Token Management
# -----------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
config_dir = os.path.join(project_root, "configs")
config_path = os.path.join(config_dir, "config.ini")

def _get_decrypted_token(key_name):
    """ 내부 공통 함수: config.ini에서 특정 key_name의 값을 읽어 복호화 """
    if not os.path.exists(config_path):
        return None

    try:
        config = ConfigObj(config_path, encoding='utf-8')
        if 'Settings' not in config:
            return None
            
        raw_token = config['Settings'].get(key_name, '')
        
        if not raw_token:
            return None
            
        try:
            decrypted = decrypt_AES(raw_token, aes_KEY, aes_IV)
            return decrypted
        except Exception:
            return raw_token # 복호화 실패 시 평문 반환
            
    except Exception as e:
        print(f"[TokenKey] Read Error ({key_name}): {e}")
        return None

def _save_encrypted_token(key_name, plain_token):
    """ 내부 공통 함수: 평문 토큰을 암호화하여 config.ini의 특정 key_name에 저장 """
    if not os.path.exists(config_dir):
        try:
            os.makedirs(config_dir)
        except Exception as e:
            return False, f"Failed to create configs directory: {e}"

    if not os.path.exists(config_path):
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                f.write("[Settings]\n")
        except Exception as e:
            return False, f"Failed to create config file: {e}"

    try:
        config = ConfigObj(config_path, encoding='utf-8')
        
        if 'Settings' not in config:
            config['Settings'] = {}
            
        if not plain_token or plain_token.strip() == "":
            config['Settings'][key_name] = ""
            msg = f"{key_name} cleared."
        else:
            encrypted = encrypt_AES(plain_token.strip(), aes_KEY, aes_IV)
            config['Settings'][key_name] = encrypted
            msg = f"{key_name} encrypted and saved."
            
        config.write()
        return True, msg
        
    except Exception as e:
        return False, str(e)

# --- 공개 API ---

def get_valid_hf_token():
    """ Hugging Face Token 조회 """
    return _get_decrypted_token('hf_token')

def save_hf_token(plain_token):
    """ Hugging Face Token 저장 """
    return _save_encrypted_token('hf_token', plain_token)

def get_valid_api_key():
    """ Google/Remote API Key 조회 """
    return _get_decrypted_token('api_key')

def save_api_key(plain_key):
    """ Google/Remote API Key 저장 """
    return _save_encrypted_token('api_key', plain_key)

# -----------------------------------------------------------------------------
# Main Test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"🔧 Token Security Manager")
    
    hf = get_valid_hf_token()
    api = get_valid_api_key()
    
    print(f"HF Token present: {bool(hf)}")
    print(f"API Key present: {bool(api)}")