from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import get_db
from app.models import User

_oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

# bcrypt itself caps secrets at 72 bytes; UserCreate already limits
# passwords to 128 characters (see app/schemas.py), so truncate defensively
# rather than let a rare multi-byte-heavy password raise at hash time.
_MAX_PASSWORD_BYTES = 72


def hash_password(password: str) -> str:
    encoded = password.encode("utf-8")[:_MAX_PASSWORD_BYTES]
    return bcrypt.hashpw(encoded, bcrypt.gensalt()).decode("ascii")


def verify_password(password: str, hashed: str) -> bool:
    encoded = password.encode("utf-8")[:_MAX_PASSWORD_BYTES]
    return bcrypt.checkpw(encoded, hashed.encode("ascii"))


def create_access_token(user_id: str) -> str:
    settings = get_settings()
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_expires_minutes)
    payload = {"sub": user_id, "exp": expires_at}
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def _decode_token(token: str) -> str:
    settings = get_settings()
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    except jwt.PyJWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")
    user_id: Optional[str] = payload.get("sub")
    if user_id is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    return user_id


def get_current_user(
    token: str = Depends(_oauth2_scheme), db: Session = Depends(get_db)
) -> User:
    user_id = _decode_token(token)
    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user
