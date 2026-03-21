"""VaultMem exceptions."""


class VaultMemError(Exception):
    """Base class for all VaultMem errors."""


class WrongPassphraseError(VaultMemError):
    """Passphrase did not match any credential slot (MEK unwrap auth tag failed)."""


class VaultTamperedError(VaultMemError):
    """GCM authentication tag verification failed — vault may have been modified."""


class VaultLockedError(VaultMemError):
    """Operation requires an open session."""


class VaultAlreadyOpenError(VaultMemError):
    """Another session holds the vault lock."""


class SessionStateError(VaultMemError):
    """Invalid session state transition."""


class MemorySchemaError(VaultMemError):
    """Memory object violates schema constraints (size limits, required fields)."""


class RotationRequiredError(VaultMemError):
    """MEK rotation is overdue and must be performed before continuing."""
