const fs = require('fs');
const path = require('path');

/**
 * Extract title from HTML <title> tag
 */
function extractTitle(content) {
    const match = content.match(/<title>(.*?)<\/title>/i);
    return match ? match[1] : 'Untitled';
}

/**
 * Extract first paragraph text
 */
function extractDescription(content) {
    const match = content.match(/<p[^>]*>(.*?)<\/p>/i);
    if (match) {
        let desc = match[1];
        // Remove HTML tags
        desc = desc.replace(/<[^>]*>/g, '');
        // Decode HTML entities
        desc = desc.replace(/&nbsp;/g, ' ').replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>');
        return desc.trim().substring(0, 120); // limit to 120 chars
    }
    return 'No description available';
}

/**
 * Scan directory for HTML files and extract metadata
 */
function scanDirectory(dirPath, { includeId = false } = {}) {
    const items = [];
    
    if (!fs.existsSync(dirPath)) {
        console.warn(`Directory not found: ${dirPath}`);
        return items;
    }

    const files = fs.readdirSync(dirPath).filter(f => f.endsWith('.html'));

    files.forEach(file => {
        const filePath = path.join(dirPath, file);
        const content = fs.readFileSync(filePath, 'utf-8');
        
        const title = extractTitle(content);
        const desc = extractDescription(content);
        // href relative to project root (index.html is at __dirname)
        const href = path.relative(__dirname, filePath).replace(/\\/g, '/');

        const item = { title, desc, href };
        if (includeId) {
            // derive id from filename without extension
            const base = path.basename(file, '.html');
            item.id = base;
        }

        items.push(item);
    });

    return items;
}

/**
 * Format items array as JavaScript code
 */
function formatItemsArray(items) {
    return items.map(item => `
            {
                title: '${escapeString(item.title)}',
                desc: '${escapeString(item.desc)}',
                href: '${escapeString(item.href)}'${item.id ? `,
                id: '${escapeString(item.id)}'` : ''}${item.tags ? `,
                tags: [${item.tags.map(t => `'${escapeString(t)}'`).join(', ')}]` : ''}
            }`).join(',');
}

/**
 * Escape single quotes in strings
 */
function escapeString(str) {
    return str.replace(/'/g, "\\'");
}

/**
 * Main build function
 */
function build() {
    console.log('🔍 Scanning pages...\n');

    // Scan directories
    const byLectureItems = scanDirectory(
        path.join(__dirname, 'pages', 'Lectures')
    );
    
    const contentDir = path.join(__dirname, 'pages', 'Content');
    const byContentItems = scanDirectory(
        contentDir
    ).map(item => {
        const base = path.basename(item.href, '.html');
        const tags = inferContentTags(base);
        return tags ? { ...item, tags } : item;
    });

    function inferContentTags(fileBase) {
        switch (fileBase) {
            case 'printing-inputs':
                return ['input', 'output', 'cout', 'cin'];
            case 'characters':
                return ['char', 'ascii', 'characters'];
            case 'math':
                return ['math', 'arithmetic', 'cmath'];
            case 'exceptions':
                return ['exceptions', 'error', 'try', 'catch'];
            case 'strings':
                return ['strings', 'string', 'text'];
            case 'vectors':
                return ['vectors', 'arrays', 'dynamic'];
            case 'time-random':
                return ['time', 'random', 'rand'];
            case 'classes':
                return ['classes', 'objects', 'oop'];
            case 'class-templates':
                return ['templates', 'classes', 'generic'];
            case 'function-templates':
                return ['templates', 'functions', 'generic'];
            default:
                return undefined;
        }
    }

    const quizDir = path.join(__dirname, 'pages', 'Quizzes');
    const byQuizItems = scanDirectory(
        quizDir,
        { includeId: true }
    ).map(item => {
        const base = path.basename(item.href, '.html');
        const inferred = inferQuizMeta(base);
        const merged = { ...item };
        if (inferred?.desc && (!merged.desc || merged.desc === 'No description available')) {
            merged.desc = inferred.desc;
        }
        if (inferred?.tags) {
            merged.tags = inferred.tags;
        }
        return merged;
    });
    
    const byExamItems = scanDirectory(
        path.join(__dirname, 'pages', 'examinations')
    );

    console.log(`✓ Found ${byLectureItems.length} lecture(s)`);
    console.log(`✓ Found ${byContentItems.length} content page(s)`);
    console.log(`✓ Found ${byQuizItems.length} quiz page(s)\n`);
    console.log(`✓ Found ${byExamItems.length} exam page(s)\n`);

    // Read index.html
    const indexPath = path.join(__dirname, 'index.html');
    let indexContent = fs.readFileSync(indexPath, 'utf-8');

    // Replace lecture items array
    const lectureArrayRegex = /const byLectureItems = \[([\s\S]*?)\];/;
    indexContent = indexContent.replace(
        lectureArrayRegex,
        `const byLectureItems = [${formatItemsArray(byLectureItems)}\n        ];`
    );

    // Replace content items array
    const contentArrayRegex = /const byContentItems = \[([\s\S]*?)\];/;
    indexContent = indexContent.replace(
        contentArrayRegex,
        `const byContentItems = [${formatItemsArray(byContentItems)}\n        ];`
    );

    const quizArrayRegex = /const byQuizItems = \[([\s\S]*?)\];/;
    indexContent = indexContent.replace(
        quizArrayRegex,
        `const byQuizItems = [${formatItemsArray(byQuizItems)}\n        ];`
    );

    // Replace exam items array
    const examArrayRegex = /const byExamItems = \[([\s\S]*?)\];/;
    indexContent = indexContent.replace(
        examArrayRegex,
        `const byExamItems = [${formatItemsArray(byExamItems)}\n        ];`
    );

    // Write back
    fs.writeFileSync(indexPath, indexContent, 'utf-8');

    console.log('✅ index.html updated successfully!');
    console.log(`   ${byLectureItems.length + byContentItems.length + byExamItems.length} total pages indexed`);
}

// Run build
build();

/**
 * Infer quiz descriptions and tags by filename base
 */
function inferQuizMeta(fileBase) {
    switch (fileBase) {
        case 'quiz-printing-inputs':
            return {
                desc: 'Test your knowledge of cout, cin, and related output functions.',
                tags: ['quiz', 'input', 'output']
            };
        case 'quiz-strings':
            return {
                desc: 'Test your knowledge of std::string and string operations.',
                tags: ['quiz', 'strings']
            };
        case 'quiz-characters':
            return {
                desc: 'Test your knowledge of char type, ASCII, and character functions.',
                tags: ['quiz', 'characters']
            };
        case 'quiz-math':
            return {
                desc: 'Test your knowledge of cmath functions and arithmetic operations.',
                tags: ['quiz', 'math']
            };
        case 'quiz-vectors':
            return {
                desc: 'Test your knowledge of std::vector and dynamic arrays.',
                tags: ['quiz', 'vectors']
            };
        case 'quiz-exceptions':
            return {
                desc: 'Test your knowledge of try, catch, throw, and exception handling.',
                tags: ['quiz', 'exceptions']
            };
        case 'quiz-time-random':
            return {
                desc: 'Test your knowledge of time operations and random number generation.',
                tags: ['quiz', 'time', 'random']
            };
        case 'quiz-classes':
            return {
                desc: 'Test your knowledge of class design, constructors, and object-oriented programming.',
                tags: ['quiz', 'classes', 'oop']
            };
        default:
            return undefined;
    }
}
