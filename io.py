from typing import Union

DEFAULT_ENCODING = 'utf8'

def decode_if_necessary(db_object: Union[str, bytes], encoding: str = DEFAULT_ENCODING) -> str:
    return (db_object.decode(encoding) if isinstance(db_object, bytes) else db_object)

def encode_if_necessary(db_object: Union[str, bytes], encoding: str = DEFAULT_ENCODING) -> bytes:
    return (db_object.encode(encoding) if isinstance(db_object, str) else db_object)

def can_encode_and_decode(db_object: Union[str, bytes], encoding: str = DEFAULT_ENCODING) -> bool:
    try:
        if isinstance(db_object, bytes):
            return (encode_if_necessary(decode_if_necessary(db_object)) == db_object)
        elif isinstance(db_object, str):
            return (decode_if_necessary(encode_if_necessary(db_object)) == db_object)
        else:
            raise Exception(type(db_object))
    except (UnicodeEncodeError, UnicodeDecodeError):
        return False

if __name__ == '__main__':
    for db_object in ['bob', b'bob', '象形字']:
        print(db_object, can_encode_and_decode(db_object))
