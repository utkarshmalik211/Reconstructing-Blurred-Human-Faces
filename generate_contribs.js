const { join } = require('path')
const { writeFile } = require('fs')
const { promisify } = require('util')
const { exec } = require('child_process')

const execAsync = promisify(exec)
const writeFileAsync = promisify(writeFile)

const AUTHOR = 'utkarshmalik211'
const EMAIL = 'utkarshmalik211@gmail.com'
const DAYS_TO_GENERATE = 180

const dateMinusGivenDays = days => {
  const d = new Date()
  d.setDate(d.getDate() - days)
  return d
}

const touchFile = async path => await writeFileAsync(path, Math.random())

const addChange = async change => await execAsync(`git add ${change}`)

const makeCommitInPast = async (date, message) =>
  await execAsync(`GIT_AUTHOR_DATE='${date}' GIT_COMMITTER_DATE='${date}' git commit --author='${AUTHOR} <${EMAIL}>' -m '${message}'`)

const applyChanges = async limit => {
  const file = join(__dirname, 'temp.txt')
  const date = dateMinusGivenDays(limit)
  await touchFile(file)
  await addChange(file)
  await makeCommitInPast(date, Math.random())
}

const execute = async count => {
  while (count--) await applyChanges(count)
}

execute(DAYS_TO_GENERATE)
