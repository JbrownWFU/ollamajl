package main

// AES (Advanced Encryption Standard) is the encryption algorithm that actually encrypts / decrypts
// GCM (Galois/Counter Mode) is a specific mode of operation for block ciphers (such as AES)
// AEAD (Authenticated Encrypted with Associated Data) is a category of encryption modes such as GCM providing data encryption & authentication

import (
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"encoding/hex"
	"flag"
	"fmt"
	"io"
	"os"

	"golang.design/x/clipboard"
)

const (
	// The keysize in bytes we are dealing with. 32 bytes = 256 bits
	KeySize = 32
	// The environment variable to be called to retrieve a stored encryption key.
	EnvKeyVar = "PKEYS_PASS"

	// Help messages
	pkeysUsage = `pkeys is a simple command-line tool for AES-256 GCM encryption.

Usage:
    pkeys <command> [flags] [arguments]

Available Commands:
    generate    	Generates a new 32-byte (256-bit) AES encryption key.
	peek			Prints the last 4 characters of the currently set environment key.
    -e, encrypt     Encrypts a plaintext string.
    -d, decrypt     Decrypts a hexadecimal ciphertext string.

Use "pkeys <command> -h" for more information about a command.
`
	generateUsage = `Usage: pkeys generate

Generates a 32 byte (256 bit) AES encryption key. The key must be set in the PKEYS_PASS
environment variable to be used for encryption / decryption.

Example:
	pkeys generate
`

	encryptUsage = `Usage: pkeys encrypt <-v> [plaintext]

Encrypts the passed plaintext string. If no plaintext string is provided, the system
clipboard contents will be encrypted and overwritten. The key must be set in the PKEYS_PASS
environment variable. Returns the encrypted plaintext as a hexadecimal string.

The verbose flag -v can be passed to print the encrypted string.

The shortcut '-e' flag can be used to call encrypt.

Example clipboard:
	pkeys encrypt

Example plaintext:
	pkeys encrypt "This is a secret!"
	
Example shortcut:
	pkeys -e
`

	decryptUsage = `Usage: pkeys decrypt <-v> [ciphertext]

Decrypts the passed hexadecimal ciphertext string. If no ciphertext string is provided, the system
clipboard contents will be decrypted and overwritten. The key must be set in the PKEYS_PASS
environment variable. Returns the unencrypted plaintext string.

The verbose flag -v can be passed to print the plaintext string.

The shortcut '-d' flag can be used to call decrypt.

Example clipboard:
	pkeys decrypt

Example ciphertext:
	pkeys decrypt b84447bfb0298cf1de9b48e92eee5fff71d10cf1ac874080632d5f11976913e1eb9054d51293dff9c9bf25e42e

Example shortcut:
	pkeys -d
`

	peekUsage = `Usage: pkeys peek

Peeks and prints the last 4 characters of the current set environment key 'PKEYS_PASS'.
If the key is not set or is of an invalid length, an error will be returned instead.

Example:
	pkeys peek
`
)

// Helper function for exiting on error
func fatalf(format string, args ...interface{}) {
	fmt.Fprintf(os.Stderr, "Error: "+format+"\n", args...)
	os.Exit(1)
}

// Helper function for retrieveing env key var
func getEnvKey() ([]byte, error) {
	keyString := os.Getenv(EnvKeyVar)
	if keyString == "" {
		return nil, fmt.Errorf(`environment key '%s' not set.

Set for this session by running:
	'set %s=d256ca4...'

Or for Powershell:
	'$env:%s="d256ca4..."'`, EnvKeyVar, EnvKeyVar, EnvKeyVar)
	}

	// Decode key into proper format
	key, err := hex.DecodeString(keyString)
	if err != nil {
		return nil, fmt.Errorf("key '%v' is not a valid hex string", err)
	}

	// Verify key length
	if len(key) != KeySize {
		return nil, fmt.Errorf("invalid key size - must be %d bytes, but got %d", KeySize, len(key))
	}

	return key, nil
}

// Returns last 4 digits of current key
func peekEnvKey() ([]byte, error) {
	// Try and get ENV key, print out last 4 chars
	currKey, err := getEnvKey()
	if err != nil {
		return nil, err
	}

	// Return last 4
	lastFourBytes := currKey[len(currKey)-4:]

	return lastFourBytes, nil
}

// Generate key for encryption
func generate() (string, error) {
	// Create empty byte slice
	key := make([]byte, KeySize)

	// Read bytes from the OS entropy source into the key byte slice
	if _, err := io.ReadFull(rand.Reader, key); err != nil {
		return "", fmt.Errorf("failed to generate random key: %w", err)
	}

	// Encode key to hex and return
	return hex.EncodeToString(key), nil
}

// Encrypt a string
func encrypt(key []byte, plaintext []byte) ([]byte, error) {
	// Define AES cipher block
	block, err := aes.NewCipher(key)
	if err != nil {
		return nil, err
	}

	// Wrap block for GCM mode
	aesgcm, err := cipher.NewGCM(block)
	if err != nil {
		return nil, err
	}

	// Get nonce for encryption of size aesgcm.NonceSize() (12 bytes)
	nonce := make([]byte, aesgcm.NonceSize())
	if _, err := io.ReadFull(rand.Reader, nonce); err != nil {
		return nil, err
	}

	// Seal plaintext
	// nonce is passed first as the destintaion, then as the nonce for appending
	ciphertext := aesgcm.Seal(nonce, nonce, plaintext, nil)

	return ciphertext, nil
}

func encryptHandler(subArgs []string) {
	// Flag package
	encryptCmd := flag.NewFlagSet("encrypt", flag.ExitOnError)

	// Verbose flag
	var verboseFlag bool

	encryptCmd.BoolVar(&verboseFlag, "v", false, "Enable verbose output.")
	encryptCmd.BoolVar(&verboseFlag, "verbose", false, "Enable verbose output.")

	// Add verbose output

	encryptCmd.Usage = func() {
		fmt.Println(encryptUsage)
		fmt.Println("\nFlags:")
		encryptCmd.PrintDefaults()
	}

	encryptCmd.Parse(subArgs)
	encryptArgs := encryptCmd.Args()

	var plaintext []byte
	var clipboardErr error

	// If no args are passed use clipboard as data source
	if len(encryptArgs) == 0 {

		plaintext, clipboardErr = getClipboard()
		if clipboardErr != nil {
			fatalf("could not retrieve clipboard contents.", clipboardErr)
		}

	} else if len(encryptArgs) == 1 {
		plaintext = []byte(encryptArgs[0])

	} else {
		fatalf(`incorrect number of arguments provided for encrypt.
		
Please provide a single plaintext string or no arguments to use the clipboard.

See 'pkeys encrypt -h' for help.`)
	}

	// Get PKEYS_PASS env val
	key, err := getEnvKey()
	if err != nil {
		fatalf("error reading key. %v\n", err)
	}

	ciphertext, err := encrypt(key, plaintext)
	if err != nil {
		fatalf("cannot encrypt messsage. %v\n", err)
	}

	// Write plaintext back to the clipboard
	// Encode to hex string -> byte slice
	clipboardCiphertext := hex.EncodeToString(ciphertext)

	writeError := writeClipboard([]byte(clipboardCiphertext))
	if writeError != nil {
		fmt.Printf("Error - unable to write clipboard contents.\nTry 'encrypt -v' for verbose output printing. %w\n", writeError)
	}

	fmt.Println("encrypted")

	// If verbose
	if verboseFlag {
		fmt.Printf("%x", ciphertext)
	}

}

// Decrypt a string
func decrypt(key []byte, ciphertext []byte) ([]byte, error) {
	block, err := aes.NewCipher(key)
	if err != nil {
		return nil, err
	}

	aesgcm, err := cipher.NewGCM(block)
	if err != nil {
		return nil, err
	}

	// Split nonce from ciphertext
	nonce, ciphertext := ciphertext[:aesgcm.NonceSize()], ciphertext[aesgcm.NonceSize():]

	plaintext, err := aesgcm.Open(nil, nonce, ciphertext, nil)
	if err != nil {
		return nil, err
	}

	return plaintext, nil
}

func decryptHandler(subArgs []string) {
	decryptCmd := flag.NewFlagSet("decrypt", flag.ExitOnError)

	// Verbose flag
	var verboseFlag bool

	decryptCmd.BoolVar(&verboseFlag, "v", false, "Enable verbose output.")
	decryptCmd.BoolVar(&verboseFlag, "verbose", false, "Enable verbose output.")

	decryptCmd.Usage = func() {
		fmt.Println(decryptUsage)
		fmt.Println("\nFlags:")
		decryptCmd.PrintDefaults()
	}

	decryptCmd.Parse(subArgs)
	decryptArgs := decryptCmd.Args()

	var ciphertextString string

	// If no other args, decrypt from clipboard, overwrite clipboard text
	if len(decryptArgs) == 0 {

		ciphertextHex, err := getClipboard()
		if err != nil {
			fatalf("could not retrieve ciphertext from clipboard.", err)
		}

		ciphertextString = string(ciphertextHex)

	} else if len(decryptArgs) == 1 {
		ciphertextString = decryptArgs[0]

	} else {
		fatalf(`incorrect number of arguments provided for decrypt.

Please provide a single hexadecimal string or no arguments to use the clipboard.

See 'pkeys decrypt -h' for help.`)
	}

	// Get key
	key, err := getEnvKey()
	if err != nil {
		fatalf("cannot encrypt messsage. %v\n", err)
	}

	ciphertext, err := hex.DecodeString(ciphertextString)
	if err != nil {
		fatalf("cannot convert ciphertext hexcode to byte slice. Please check the format of your ciphertext. %v\n", err)
	}

	plaintext, err := decrypt(key, ciphertext)
	if err != nil {
		fatalf("cannot decrypt ciphertext: %v\n(This often means the key is incorrect or the ciphertext is corrupt.)", err)
	}

	clipboard.Write(clipboard.FmtText, []byte(plaintext))

	// Feedback success
	fmt.Println("decrypted")

	// Print to terminal if needed
	if verboseFlag {
		fmt.Printf("%s", plaintext)
	}

}

// Returns clipboard contents
func getClipboard() ([]byte, error) {
	err := clipboard.Init()
	if err != nil {
		return nil, fmt.Errorf("could not initialize clipboard. %w", err)
	}

	clipboardContents := clipboard.Read(clipboard.FmtText)

	if len(clipboardContents) == 0 {
		return nil, fmt.Errorf("clipboard is empty. %w", err)
	}

	return clipboardContents, nil
}

// Write clipboard contents
func writeClipboard(toWrite []byte) error {
	err := clipboard.Init()
	if err != nil {
		return err
	}

	clipboard.Write(clipboard.FmtText, toWrite)
	return nil
}

func main() {
	args := os.Args

	if len(args) < 2 {
		fatalf(`no command specified.

See 'pkeys -h' for usage information.`)
	}

	// Get called command
	cmd := args[1]
	subArgs := args[2:]

	switch cmd {

	case "encrypt", "-e":
		encryptHandler(subArgs)

	case "decrypt", "-d":
		decryptHandler(subArgs)
	
	case "help", "-h":
		fmt.Println(pkeysUsage)

	case "peek":
		statusCmd := flag.NewFlagSet("peek", flag.ExitOnError)

		statusCmd.Usage = func() {
			fmt.Println(peekUsage)
		}

		statusCmd.Parse(subArgs)
		statusArgs := statusCmd.Args()

		if len(statusArgs) != 0 {
			fatalf(`incorrect number of arguments passed to peek.
			
See 'peek -h' for help.`)
		}

		peek, err := peekEnvKey()
		if err != nil {
			fatalf("could not peek environment key. %v\n", err)
		}

		peekString := hex.EncodeToString(peek)
		fmt.Printf("Last 4 of current key: %s", peekString)

	case "generate":
		generateCmd := flag.NewFlagSet("generate", flag.ExitOnError)

		generateCmd.Usage = func() {
			fmt.Println(generateUsage)
		}

		generateCmd.Parse(subArgs)

		args := generateCmd.Args()
		if len(args) != 0 {
			fatalf(`incorrect number of arguments passed to generate.

See 'pkeys generate -h' for help.`)
		}

		key, err := generate()
		if err != nil {
			fatalf("cannot generate key. %v\n", err)
		}

		fmt.Println("Generated AES-256 Key:")
		fmt.Printf("%s\n", key)


	default:
		fatalf("unknown command '%s'\nPkeys Usage: generate | encrypt <-v> [plaintext] | decrypt <-v> [ciphertext]", cmd)
	}
}