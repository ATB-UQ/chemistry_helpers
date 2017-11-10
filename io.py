from typing import Union

DEFAULT_ENCODING = 'utf8'

def decode_if_necessary(db_object: Union[str, bytes], encoding: str = DEFAULT_ENCODING, errors: str = 'strict') -> str:
    return (db_object.decode(encoding, errors=errors) if isinstance(db_object, bytes) else db_object)

def encode_if_necessary(db_object: Union[str, bytes], encoding: str = DEFAULT_ENCODING, errors: str = 'strict') -> bytes:
    return (db_object.encode(encoding, errors=errors) if isinstance(db_object, str) else db_object)

def can_encode_and_decode(db_object: Union[str, bytes], encoding: str = DEFAULT_ENCODING, errors: str = 'strict') -> bool:
    try:
        if isinstance(db_object, bytes):
            return (encode_if_necessary(decode_if_necessary(db_object, errors=errors), errors=errors) == db_object)
        elif isinstance(db_object, str):
            return (decode_if_necessary(encode_if_necessary(db_object, errors=errors), errors=errors) == db_object)
        else:
            raise Exception(type(db_object))
    except (UnicodeEncodeError, UnicodeDecodeError):
        return False

if __name__ == '__main__':
    for db_object in ['bob', b'bob', '象形字']:
        print(db_object, can_encode_and_decode(db_object))
