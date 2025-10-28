// C++ Parser for Farm Drone Automation
// Parses simplified C++ code and extracts executable commands

class CppParser {
    constructor() {
        this.commands = [];
        this.errors = [];
        this.loopFunction = null;
    }

    parse(code) {
        this.commands = [];
        this.errors = [];
        this.loopFunction = null;

        try {
            // Remove comments
            code = this.removeComments(code);

            // Extract the loop function
            this.extractLoopFunction(code);

            if (!this.loopFunction) {
                this.errors.push("No loop() function found. Your program must have a void loop() { ... } function.");
                return { success: false, errors: this.errors };
            }

            // Parse the loop function body
            const instructions = this.parseInstructions(this.loopFunction);

            return {
                success: this.errors.length === 0,
                instructions: instructions,
                errors: this.errors
            };
        } catch (error) {
            this.errors.push(`Parse error: ${error.message}`);
            return { success: false, errors: this.errors };
        }
    }

    removeComments(code) {
        // Remove single-line comments
        code = code.replace(/\/\/.*/g, '');
        // Remove multi-line comments
        code = code.replace(/\/\*[\s\S]*?\*\//g, '');
        return code;
    }

    extractLoopFunction(code) {
        // Find void loop() { ... }
        const loopRegex = /void\s+loop\s*\(\s*\)\s*\{([\s\S]*)\}/;
        const match = code.match(loopRegex);
        
        if (match) {
            this.loopFunction = match[1];
        }
    }

    parseInstructions(code) {
        const instructions = [];
        
        // Tokenize the code
        const tokens = this.tokenize(code);
        
        // Parse tokens into instruction tree
        const ast = this.buildAST(tokens);
        
        return ast;
    }

    tokenize(code) {
        const tokens = [];
        const tokenPatterns = [
            { type: 'KEYWORD', regex: /^(if|else|while|for|return)/ },
            { type: 'FUNCTION', regex: /^(moveForward|turnRight|turnLeft|plant|harvest|water|scan|scanAhead)\s*\(/ },
            { type: 'CONSTANT', regex: /^(EMPTY|PLANTED|GROWING|GROWN|OBSTACLE)/ },
            { type: 'COMPARISON', regex: /^(==|!=|<=|>=|<|>)/ },
            { type: 'LOGIC', regex: /^(&&|\|\|)/ },
            { type: 'NUMBER', regex: /^(\d+)/ },
            { type: 'LBRACE', regex: /^\{/ },
            { type: 'RBRACE', regex: /^\}/ },
            { type: 'LPAREN', regex: /^\(/ },
            { type: 'RPAREN', regex: /^\)/ },
            { type: 'SEMICOLON', regex: /^;/ },
            { type: 'WHITESPACE', regex: /^\s+/ },
        ];

        let remaining = code;
        while (remaining.length > 0) {
            let matched = false;

            for (const pattern of tokenPatterns) {
                const match = remaining.match(pattern.regex);
                if (match) {
                    if (pattern.type !== 'WHITESPACE') {
                        tokens.push({
                            type: pattern.type,
                            value: match[0].replace(/\s*\(\s*$/, '').trim()
                        });
                    }
                    remaining = remaining.slice(match[0].length);
                    matched = true;
                    break;
                }
            }

            if (!matched) {
                // Skip unknown character
                remaining = remaining.slice(1);
            }
        }

        return tokens;
    }

    buildAST(tokens) {
        const ast = [];
        let i = 0;

        const parseBlock = () => {
            const block = [];
            
            while (i < tokens.length) {
                const token = tokens[i];

                if (token.type === 'RBRACE') {
                    break;
                }

                if (token.type === 'FUNCTION') {
                    block.push({
                        type: 'call',
                        function: token.value
                    });
                    i++;
                    // Skip parentheses and semicolon
                    while (i < tokens.length && tokens[i].type !== 'SEMICOLON') {
                        i++;
                    }
                    if (i < tokens.length && tokens[i].type === 'SEMICOLON') {
                        i++;
                    }
                } else if (token.type === 'KEYWORD') {
                    if (token.value === 'if') {
                        i++;
                        // Parse condition
                        const condition = this.parseCondition(tokens, i);
                        i = condition.nextIndex;

                        // Parse if block
                        if (i < tokens.length && tokens[i].type === 'LBRACE') {
                            i++;
                            const ifBlock = parseBlock();
                            
                            // Check for else
                            let elseBlock = null;
                            if (i < tokens.length && tokens[i].type === 'KEYWORD' && tokens[i].value === 'else') {
                                i++;
                                if (i < tokens.length && tokens[i].type === 'LBRACE') {
                                    i++;
                                    elseBlock = parseBlock();
                                }
                            }

                            block.push({
                                type: 'if',
                                condition: condition.ast,
                                ifBlock: ifBlock,
                                elseBlock: elseBlock
                            });
                        }
                    } else if (token.value === 'while') {
                        i++;
                        const condition = this.parseCondition(tokens, i);
                        i = condition.nextIndex;

                        if (i < tokens.length && tokens[i].type === 'LBRACE') {
                            i++;
                            const whileBlock = parseBlock();

                            block.push({
                                type: 'while',
                                condition: condition.ast,
                                block: whileBlock
                            });
                        }
                    } else {
                        i++;
                    }
                } else {
                    i++;
                }
            }

            if (i < tokens.length && tokens[i].type === 'RBRACE') {
                i++;
            }

            return block;
        };

        ast.push(...parseBlock());
        return ast;
    }

    parseCondition(tokens, startIndex) {
        let i = startIndex;
        const condition = { left: null, operator: null, right: null };

        // Skip LPAREN
        if (i < tokens.length && tokens[i].type === 'LPAREN') {
            i++;
        }

        // Get left side (should be a function call like scan())
        if (i < tokens.length && tokens[i].type === 'FUNCTION') {
            condition.left = { type: 'call', function: tokens[i].value };
            i++;
            // Skip parentheses
            while (i < tokens.length && tokens[i].type !== 'COMPARISON') {
                i++;
            }
        }

        // Get operator
        if (i < tokens.length && tokens[i].type === 'COMPARISON') {
            condition.operator = tokens[i].value;
            i++;
        }

        // Get right side (constant or number)
        if (i < tokens.length) {
            if (tokens[i].type === 'CONSTANT') {
                condition.right = { type: 'constant', value: tokens[i].value };
            } else if (tokens[i].type === 'NUMBER') {
                condition.right = { type: 'number', value: parseInt(tokens[i].value) };
            }
            i++;
        }

        // Skip RPAREN
        while (i < tokens.length && tokens[i].type === 'RPAREN') {
            i++;
        }

        return { ast: condition, nextIndex: i };
    }

    // Helper method to validate API functions
    static isValidFunction(functionName) {
        const validFunctions = [
            'moveForward', 'turnRight', 'turnLeft',
            'plant', 'harvest', 'water',
            'scan', 'scanAhead'
        ];
        return validFunctions.includes(functionName);
    }

    static isValidConstant(constantName) {
        const validConstants = ['EMPTY', 'PLANTED', 'GROWING', 'GROWN', 'OBSTACLE'];
        return validConstants.includes(constantName);
    }
}

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = CppParser;
}
